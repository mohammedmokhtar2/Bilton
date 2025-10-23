#!/usr/bin/env python3
"""
Vector of Victors — v19

This solver mirrors the robust, working logic of v9 with:
- Optional clustering (NumPy/SciPy fallbacks)
- High-throughput, inventory-aware packing
- Transactional route planning with pickups-first routing
- Home node resolution that handles both warehouse IDs and node IDs
- Vehicle max_distance validation
- Correct solution format keys: routes -> [ { vehicle_id, steps: [ { node_id, pickups, deliveries, unloads } ] } ]
"""

import heapq
from typing import Dict, Iterable, List, Optional, Tuple

# Optional scientific deps: fall back gracefully if unavailable
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency for clustering only
    np = None  # type: ignore

try:
    from scipy.cluster.vq import kmeans, vq  # type: ignore
except Exception:  # pragma: no cover - optional dependency for clustering only
    kmeans = None  # type: ignore
    vq = None  # type: ignore

from robin_logistics import LogisticsEnvironment


def solver(env) -> Dict[str, List[Dict]]:
    """
    Main solver function.
    Returns solution with the required keys per API reference.
    """

    # Make stochastic components reproducible when numpy exists
    if np is not None:
        np.random.seed(42)

    solution: Dict[str, List[Dict]] = {"routes": []}

    order_ids: List[str] = env.get_all_order_ids()
    vehicle_ids: List[str] = env.get_available_vehicles()

    if not order_ids or not vehicle_ids:
        return solution

    # --- Helper Functions (Weight/Volume) ---

    def get_order_weight(order_id: str) -> float:
        """Calculate the total weight for a given order using SKU details."""
        requirements = env.get_order_requirements(order_id)
        total_weight = 0.0
        for sku_id, quantity in requirements.items():
            sku_details = env.get_sku_details(sku_id)
            if sku_details:
                sku_weight = sku_details.get("weight", 0.0)
                total_weight += sku_weight * quantity
        return total_weight

    def get_order_volume(order_id: str) -> float:
        """Calculate the total volume for a given order using SKU details."""
        requirements = env.get_order_requirements(order_id)
        total_volume = 0.0
        for sku_id, quantity in requirements.items():
            sku_details = env.get_sku_details(sku_id)
            if sku_details:
                sku_volume = sku_details.get("volume", 0.0)
                total_volume += sku_volume * quantity
        return total_volume

    order_weights = {oid: get_order_weight(oid) for oid in order_ids}
    order_volumes = {oid: get_order_volume(oid) for oid in order_ids}
    all_vehicles = [env.get_vehicle_by_id(vid) for vid in vehicle_ids]

    # --- Road Network and Dijkstra Pathfinding ---

    road_data = env.get_road_network_data() or {}
    raw_adj = road_data.get("adjacency_list", {})

    adjacency: Dict[int, List[Tuple[int, float]]] = {}

    def _normalize_neighbors(entries: Iterable):
        for entry in entries:
            if isinstance(entry, dict):
                for dst, weight in entry.items():
                    yield int(dst), float(weight)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                yield int(entry[0]), float(entry[1])
            elif isinstance(entry, (int, float)):
                yield int(entry), 1.0

    for node_key, neighbors in raw_adj.items():
        node_id = int(node_key)
        if isinstance(neighbors, dict):
            adjacency[node_id] = [(int(dst), float(weight)) for dst, weight in neighbors.items()]
        else:
            adjacency[node_id] = list(_normalize_neighbors(neighbors))

    def shortest_path(start: int, goal: int) -> Tuple[Optional[List[int]], float]:
        if start == goal:
            return [start], 0.0
        if start not in adjacency:
            return None, float("inf")

        dist: Dict[int, float] = {start: 0.0}
        parent: Dict[int, Optional[int]] = {start: None}
        pq: List[tuple[float, int]] = [(0.0, start)]
        visited: set[int] = set()

        while pq:
            cost, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == goal:
                path = []
                curr: Optional[int] = goal
                while curr is not None:
                    path.append(curr)
                    curr = parent.get(curr)
                path.reverse()
                return path, cost
            for dst, weight in adjacency.get(node, []):
                if dst in visited:
                    continue
                new_cost = cost + weight
                if new_cost < dist.get(dst, float("inf")):
                    dist[dst] = new_cost
                    parent[dst] = node
                    heapq.heappush(pq, (new_cost, dst))
        return None, float('inf')

    # --- End Pathfinding ---

    # --- K-MEANS CLUSTERING (used to guide packing) ---

    order_locations: Dict[str, List[float]] = {}
    order_nodes: Dict[str, Optional[int]] = {}
    for oid in order_ids:
        node_id = env.get_order_location(oid)
        order_nodes[oid] = node_id
        node = env.nodes.get(node_id)
        if node:
            order_locations[oid] = [node.lat, node.lon]

    clusterable_order_ids = list(order_locations.keys())

    order_to_cluster_map: Dict[str, int] = {}
    clusters_to_orders_map: Dict[int, List[str]] = {}

    if not clusterable_order_ids:
        print("Error: No order locations. Clustering failed.")
        clusters_to_orders_map[0] = list(order_ids)
        for oid in order_ids:
            order_to_cluster_map[oid] = 0
    else:
        # Perform clustering only if numpy and scipy are available
        if np is not None and kmeans is not None and vq is not None:
            locations_array = np.array([order_locations[oid] for oid in clusterable_order_ids])

            N_CLUSTERS_TO_TRY = 6
            n_clusters = min(N_CLUSTERS_TO_TRY, len(clusterable_order_ids))

            if n_clusters > 0:
                locations_array_float = locations_array.astype(float)
                try:
                    code_book, _ = kmeans(locations_array_float, n_clusters, iter=20)
                    cluster_labels, _ = vq(locations_array_float, code_book)

                    clusters_to_orders_map = {i: [] for i in range(n_clusters)}
                    for i, order_id in enumerate(clusterable_order_ids):
                        cluster_id = int(cluster_labels[i])
                        order_to_cluster_map[order_id] = cluster_id
                        clusters_to_orders_map[cluster_id].append(order_id)
                except Exception as e:
                    print(f"KMeans (scipy) failed: {e}. Using single cluster.")
                    clusters_to_orders_map[0] = list(order_ids)
                    for oid in order_ids:
                        order_to_cluster_map[oid] = 0
            else:
                clusters_to_orders_map[0] = list(order_ids)
                for oid in order_ids:
                    order_to_cluster_map[oid] = 0
        else:
            # Fallback: no clustering, single cluster
            clusters_to_orders_map[0] = list(order_ids)
            for oid in order_ids:
                order_to_cluster_map[oid] = 0

    # --- End Clustering ---

    # --- Home node resolution (handles both node-id and warehouse-id returns) ---
    def resolve_home_node_id(vehicle_id: str) -> Optional[int]:
        """
        Robustly resolve the node ID of the vehicle's home warehouse.

        Some environments return the home warehouse as a warehouse ID (str), others
        may return a node ID (int) directly. This function normalizes to a node ID.
        """
        try:
            home_ref = env.get_vehicle_home_warehouse(vehicle_id)
        except Exception:
            home_ref = None

        # If it's already an int and exists as a node, use it directly
        if isinstance(home_ref, int) and home_ref in getattr(env, 'nodes', {}):
            return home_ref

        # If it's a string (likely a warehouse ID), resolve to the warehouse's node
        if isinstance(home_ref, str):
            try:
                wh = env.get_warehouse_by_id(home_ref) if hasattr(env, 'get_warehouse_by_id') else env.warehouses.get(home_ref)
                if wh and getattr(wh, 'location', None):
                    return getattr(wh.location, 'id', None)
            except Exception:
                pass

        # Try via the vehicle object
        try:
            vehicle_obj = env.get_vehicle_by_id(vehicle_id)
            wh_id = getattr(vehicle_obj, 'home_warehouse_id', None)
            if wh_id:
                wh = env.get_warehouse_by_id(wh_id) if hasattr(env, 'get_warehouse_by_id') else env.warehouses.get(wh_id)
                if wh and getattr(wh, 'location', None):
                    return getattr(wh.location, 'id', None)
        except Exception:
            pass

        # Last resort: None (caller should handle)
        return None

    # --- Helper to generate a route (transactional, pickups-first) ---
    def generate_route_for_orders(
        vehicle_id: str,
        home_node: int,
        vehicle_orders: List[str],
        current_planning_inventory: Dict[str, Dict[str, int]]
    ) -> Tuple[Optional[List[Dict]], Dict[str, Dict[str, int]]]:
        """
        Generates and validates a route for a given list of orders using a transactional
        planning inventory. Includes vehicle.max_distance check and robust order/node handling.
        """
        # Vehicle object for max_distance
        vehicle = env.get_vehicle_by_id(vehicle_id)
        if not vehicle:
            print(f"Error: Could not find vehicle {vehicle_id}. Route failed.")
            return None, current_planning_inventory

        planning_inventory = {
            wh_id: inv.copy() for wh_id, inv in current_planning_inventory.items()
        }

        # Aggregate all order SKU requirements
        all_requirements: Dict[str, int] = {}
        for order_id in vehicle_orders:
            try:
                requirements = env.get_order_requirements(order_id)
            except KeyError:
                print(f"Warning: Order ID {order_id} not found in env.orders during route gen. Skipping.")
                return None, current_planning_inventory
            for sku, qty in requirements.items():
                all_requirements[sku] = all_requirements.get(sku, 0) + qty

        # Determine pickup warehouses ranked by path distance from home_node
        temp_pickup_stops: Dict[int, Dict[str, Dict]] = {}

        for sku_id, total_qty_needed in all_requirements.items():
            remaining_qty_to_find = total_qty_needed

            warehouses_with_item = [
                wh_id for wh_id, inv in planning_inventory.items()
                if inv.get(sku_id, 0) > 0
            ]

            if not warehouses_with_item:
                print(f"FATAL (Planning): No warehouse has any {sku_id}. Route failed for {vehicle_id}.")
                return None, current_planning_inventory

            ranked_warehouses: List[Tuple[float, str, int]] = []
            for wh_id in warehouses_with_item:
                warehouse_obj = env.get_warehouse_by_id(wh_id)
                if not warehouse_obj:
                    continue
                wh_node = warehouse_obj.location.id
                path, dist = shortest_path(home_node, wh_node)
                if path:
                    ranked_warehouses.append((float(dist), wh_id, wh_node))

            ranked_warehouses.sort()

            for _, wh_id, wh_node in ranked_warehouses:
                if remaining_qty_to_find == 0:
                    break

                available_qty = planning_inventory[wh_id].get(sku_id, 0)
                qty_to_take = min(remaining_qty_to_find, available_qty)

                if qty_to_take > 0:
                    if wh_node not in temp_pickup_stops:
                        temp_pickup_stops[wh_node] = {}

                    if sku_id not in temp_pickup_stops[wh_node]:
                        temp_pickup_stops[wh_node][sku_id] = {
                            "warehouse_id": wh_id, "sku_id": sku_id, "quantity": qty_to_take
                        }
                    else:
                        temp_pickup_stops[wh_node][sku_id]["quantity"] += qty_to_take

                    remaining_qty_to_find -= qty_to_take
                    planning_inventory[wh_id][sku_id] -= qty_to_take

            if remaining_qty_to_find > 0:
                print(
                    f"FATAL: Not enough TOTAL inventory for {sku_id} for route {vehicle_id}. "
                    f"Needed {total_qty_needed}, found {total_qty_needed - remaining_qty_to_find}. Route failed."
                )
                return None, current_planning_inventory

        pickup_stops: Dict[int, List[Dict]] = {}
        for wh_node, sku_ops in temp_pickup_stops.items():
            pickup_stops[wh_node] = list(sku_ops.values())

        # Build delivery stops and filter invalid orders with missing nodes
        delivery_stops: Dict[int, List[Dict]] = {}
        valid_vehicle_orders: List[str] = []
        for order_id in vehicle_orders:
            order_node = order_nodes.get(order_id)
            if order_node is None:
                print(f"Warning: Node not found for order {order_id} during stop creation. Skipping this order.")
                continue
            valid_vehicle_orders.append(order_id)

            if order_node not in delivery_stops:
                delivery_stops[order_node] = []

            requirements = env.get_order_requirements(order_id)
            for sku, qty in requirements.items():
                delivery_op = {"order_id": order_id, "sku_id": sku, "quantity": qty}
                delivery_stops[order_node].append(delivery_op)

        if not valid_vehicle_orders or not delivery_stops:
            print(f"Warning: No valid delivery stops for vehicle {vehicle_id}. Route failed.")
            return None, current_planning_inventory

        # Greedy pickups-first, then deliveries routing w/ shortest paths
        steps: List[Dict] = []
        current_node = home_node
        steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})

        pickup_nodes_to_visit = list(pickup_stops.keys())
        while pickup_nodes_to_visit:
            closest_node, shortest_path_to_node, min_dist = None, None, float('inf')
            for node in pickup_nodes_to_visit:
                path, dist = shortest_path(current_node, node)
                if path and dist < min_dist:
                    min_dist, closest_node, shortest_path_to_node = dist, node, path

            if closest_node is None:
                print(f"Warning: Cannot find path to pickup node from {current_node} for {vehicle_id}. Route failed.")
                return None, current_planning_inventory

            if len(shortest_path_to_node) > 1:
                for node_id in shortest_path_to_node[1:-1]:
                    steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
            steps.append({"node_id": closest_node, "pickups": pickup_stops.get(closest_node, []), "deliveries": [], "unloads": []})
            current_node = closest_node
            pickup_nodes_to_visit.remove(closest_node)

        delivery_nodes_to_visit = list(delivery_stops.keys())
        while delivery_nodes_to_visit:
            closest_node, shortest_path_to_node, min_dist = None, None, float('inf')
            for node in delivery_nodes_to_visit:
                path, dist = shortest_path(current_node, node)
                if path and dist < min_dist:
                    min_dist, closest_node, shortest_path_to_node = dist, node, path

            if closest_node is None:
                print(f"Warning: Cannot find path to delivery node from {current_node} for {vehicle_id}. Route failed.")
                return None, current_planning_inventory

            if len(shortest_path_to_node) > 1:
                for node_id in shortest_path_to_node[1:-1]:
                    steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
            steps.append({"node_id": closest_node, "pickups": [], "deliveries": delivery_stops.get(closest_node, []), "unloads": []})
            current_node = closest_node
            delivery_nodes_to_visit.remove(closest_node)

        path_home_list, _ = shortest_path(current_node, home_node)
        if not path_home_list:
            print(f"Warning: Cannot find path home from {current_node} for {vehicle_id}. Route failed.")
            return None, current_planning_inventory

        if len(path_home_list) > 1:
            for node_id in path_home_list[1:-1]:
                steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
        steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})

        # --- MAX DISTANCE CHECK ---
        route_node_ids = [step['node_id'] for step in steps]
        total_route_distance = env.get_route_distance(route_node_ids)

        if total_route_distance is None:
            print(f"Warning: Could not calculate route distance for {vehicle_id}. Route failed.")
            return None, current_planning_inventory

        max_dist = getattr(vehicle, 'max_distance', None)
        if isinstance(max_dist, (int, float)) and max_dist > 0 and total_route_distance > max_dist:
            print(
                f"Validation FAILED for {vehicle_id}: Route distance ({total_route_distance:.2f} km) "
                f"exceeds vehicle max distance ({max_dist} km). Returning None."
            )
            return None, current_planning_inventory
        # --- END MAX DISTANCE CHECK ---

        is_valid, msg = env.validator.validate_route_steps(vehicle_id, steps)
        if is_valid:
            return steps, planning_inventory
        else:
            print(f"Validation FAILED for {vehicle_id} (after distance check): {msg}. Returning None.")
            return None, current_planning_inventory

    # --- MAIN LOOP: High-Throughput Inventory-Aware Greedy Bin Packing ---

    planning_inventory = {
        wh_id: wh.inventory.copy() for wh_id, wh in env.warehouses.items()
    }

    total_planning_inventory: Dict[str, int] = {}
    for wh_id, inv in planning_inventory.items():
        for sku, qty in inv.items():
            total_planning_inventory[sku] = total_planning_inventory.get(sku, 0) + qty

    all_vehicles.sort(key=lambda v: (v.capacity_weight, v.capacity_volume), reverse=True)
    unassigned_order_set = set(order_ids)
    used_vehicles_in_main_loop: set[str] = set()

    for vehicle in all_vehicles:
        if not unassigned_order_set:
            break

        vehicle_orders: List[str] = []
        current_weight = 0.0
        current_volume = 0.0
        tentative_requirements: Dict[str, int] = {}

        orders_to_check = sorted(
            list(unassigned_order_set),
            key=lambda oid: order_to_cluster_map.get(oid, -1)
        )

        for order_id in orders_to_check:

            order_weight = order_weights.get(order_id)
            order_volume = order_volumes.get(order_id)
            if order_weight is None or order_volume is None:
                print(f"Warning: Missing weight/volume for {order_id}. Skipping.")
                continue

            if (current_weight + order_weight > vehicle.capacity_weight or
                current_volume + order_volume > vehicle.capacity_volume):
                continue

            order_reqs = env.get_order_requirements(order_id)
            is_possible = True

            for sku, qty in order_reqs.items():
                current_tentative_needed = tentative_requirements.get(sku, 0)
                new_total_tentative_needed = current_tentative_needed + qty
                total_available_in_map = total_planning_inventory.get(sku, 0)

                if new_total_tentative_needed > total_available_in_map:
                    is_possible = False
                    if not vehicle_orders:
                        print(
                            f"Warning (Pre-check): Cannot pack {order_id} onto {vehicle.id}. "
                            f"Insufficient total inventory for {sku}."
                        )
                    break

            if is_possible:
                vehicle_orders.append(order_id)
                current_weight += order_weight
                current_volume += order_volume
                unassigned_order_set.remove(order_id)

                for sku, qty in order_reqs.items():
                    tentative_requirements[sku] = tentative_requirements.get(sku, 0) + qty
            else:
                continue

        if vehicle_orders:
            vehicle_id = vehicle.id
            home_node = resolve_home_node_id(vehicle_id)
            if home_node is None:
                print(f"Error: Could not resolve home node for {vehicle_id}. Returning orders to pool.")
                unassigned_order_set.update(vehicle_orders)
                continue

            steps, new_planning_inventory = generate_route_for_orders(
                vehicle_id, home_node, vehicle_orders, planning_inventory
            )

            if steps:
                solution["routes"].append({"vehicle_id": vehicle_id, "steps": steps})
                planning_inventory = new_planning_inventory
                used_vehicles_in_main_loop.add(vehicle_id)

                total_planning_inventory = {}
                for wh_id, inv in planning_inventory.items():
                    for sku, qty in inv.items():
                        total_planning_inventory[sku] = total_planning_inventory.get(sku, 0) + qty
            else:
                print(f"Main route gen failed for {vehicle_id}. Returning {len(vehicle_orders)} orders to pool.")
                unassigned_order_set.update(vehicle_orders)

    # --- FINAL FALLBACK LOOP: Single Order Per Vehicle ---
    if unassigned_order_set:
        print(f"\nEntering final fallback loop for {len(unassigned_order_set)} orders.")

        available_fallback_vehicles = [
            v for v in all_vehicles if v.id not in used_vehicles_in_main_loop
        ]
        available_fallback_vehicles.sort(key=lambda v: (v.capacity_weight, v.capacity_volume))

        fallback_orders_to_assign = sorted(list(unassigned_order_set))
        assigned_in_fallback: set[str] = set()

        for order_id in fallback_orders_to_assign:
            assigned_this_order = False
            for vehicle_idx in range(len(available_fallback_vehicles)):
                vehicle = available_fallback_vehicles[vehicle_idx]
                vehicle_id = vehicle.id

                order_weight = order_weights.get(order_id)
                order_volume = order_volumes.get(order_id)
                if order_weight is None or order_volume is None:
                    continue

                if (order_weight <= vehicle.capacity_weight and
                    order_volume <= vehicle.capacity_volume):

                    home_node = resolve_home_node_id(vehicle_id)
                    if home_node is None:
                        print(f"Error: Could not resolve home node for {vehicle_id} in fallback.")
                        continue

                    steps, new_planning_inventory = generate_route_for_orders(
                        vehicle_id, home_node, [order_id], planning_inventory
                    )

                    if steps:
                        print(f"Fallback success: Assigned {order_id} to {vehicle_id}.")
                        solution["routes"].append({"vehicle_id": vehicle_id, "steps": steps})
                        planning_inventory = new_planning_inventory
                        assigned_in_fallback.add(order_id)
                        assigned_this_order = True

                        # Update total inventory
                        total_planning_inventory = {}
                        for wh_id, inv in planning_inventory.items():
                            for sku, qty in inv.items():
                                total_planning_inventory[sku] = total_planning_inventory.get(sku, 0) + qty

                        # Remove this vehicle from further fallback consideration
                        available_fallback_vehicles.pop(vehicle_idx)
                        break

            if not assigned_this_order:
                print(f"Fallback failed for order {order_id}: No suitable vehicle or valid route found.")

        unassigned_order_set -= assigned_in_fallback

    if unassigned_order_set:
        print(f"\nWarning: {len(unassigned_order_set)} orders were truly unassigned.")

    return solution


if __name__ == "__main__":
    print("Running solver v19 locally...")
    env = LogisticsEnvironment()

    # Optional: load a specific scenario here if desired

    env.set_solver(solver)

    # To run headlessly:
    print("Running headless...")
    results = env.run_headless("vector_of_victors_solver_v19_run")
    print("\n--- Solver Results ---")
    print(results)

    # Reset environment state for a clean stats run
    print("\nResetting environment state for statistics run...")
    env.reset_all_state()

    # Second run for statistics
    print("Running solver again for stats...")
    solution = solver(env)
    stats = env.get_solution_statistics(solution)
    print("\n--- Solution Statistics (from fresh run) ---")
    print(stats)
