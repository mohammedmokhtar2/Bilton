#!/usr/bin/env python3
"""
Final Optimized Solver for the Beltone AI Hackathon.

This solver uses "High-Throughput Bin Packing" (Weight/Volume only)
and trusts the route generator's "Transactional Inventory" to
validate and commit routes. This prevents "all-or-nothing" failures.
"""

import heapq
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from scipy.cluster.vq import kmeans, vq  # Use Scipy for clustering

from robin_logistics import LogisticsEnvironment


def solver(env) -> Dict[str, List[Dict]]:
    """
    Main solver function.
    """
    
    np.random.seed(42)
    
    solution: Dict[str, List[Dict]] = {"routes": []}

    order_ids: List[str] = env.get_all_order_ids()
    vehicle_ids: List[str] = env.get_available_vehicles()
    
    if not order_ids or not vehicle_ids:
        return solution 

    # --- Helper Functions (Weight/Volume) ---
    
    def get_order_weight(order_id: str) -> float:
        requirements = env.get_order_requirements(order_id)
        total_weight = 0.0
        for sku_id, quantity in requirements.items():
            sku_details = env.get_sku_details(sku_id)
            if sku_details:
                sku_weight = sku_details.get("weight", 0.0)
                total_weight += sku_weight * quantity
        return total_weight

    def get_order_volume(order_id: str) -> float:
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
            return None, float('inf') 

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
    
    order_locations = {}
    order_nodes = {} 
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
        locations_array = np.array([order_locations[oid] for oid in clusterable_order_ids])
        
        N_CLUSTERS_TO_TRY = 6 
        n_clusters = min(N_CLUSTERS_TO_TRY, len(clusterable_order_ids))
        
        if n_clusters > 0:
            locations_array_float = locations_array.astype(float)
            try:
                code_book, distortion = kmeans(locations_array_float, n_clusters, iter=20)
                cluster_labels, dist_to_centers = vq(locations_array_float, code_book)
                
                clusters_to_orders_map = {i: [] for i in range(n_clusters)}
                for i, order_id in enumerate(clusterable_order_ids):
                    cluster_id = cluster_labels[i]
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

    # --- End Clustering ---


    # ---
    # Helper function to generate a route.
    # ---
    def generate_route_for_orders(
        vehicle_id: str, 
        home_node: int, 
        vehicle_orders: List[str],
        current_planning_inventory: Dict[str, Dict[str, int]]
    ) -> Tuple[Optional[List[Dict]], Dict[str, Dict[str, int]]]:
        """
        Generates and validates a route for a given list of orders
        using a *transactional* planning inventory.
        """
        
        planning_inventory = { 
            wh_id: inv.copy() for wh_id, inv in current_planning_inventory.items()
        }

        all_requirements = {}
        for order_id in vehicle_orders:
            requirements = env.get_order_requirements(order_id)
            for sku, qty in requirements.items():
                all_requirements[sku] = all_requirements.get(sku, 0) + qty
        
        temp_pickup_stops: Dict[int, Dict[str, Dict]] = {}
        
        for sku_id, total_qty_needed in all_requirements.items():
            remaining_qty_to_find = total_qty_needed
            
            warehouses_with_item = [
                wh_id for wh_id, inv in planning_inventory.items() 
                if inv.get(sku_id, 0) > 0
            ]
            
            if not warehouses_with_item:
                print(f"FATAL (Planning): No warehouse has any {sku_id}. Route failed.")
                return None, current_planning_inventory 
            
            ranked_warehouses = []
            for wh_id in warehouses_with_item:
                wh_node = env.get_warehouse_by_id(wh_id).location.id
                path, dist = shortest_path(home_node, wh_node)
                if path:
                    ranked_warehouses.append((dist, wh_id, wh_node))
            
            ranked_warehouses.sort() 
            
            for dist, wh_id, wh_node in ranked_warehouses:
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
                print(f"FATAL: Not enough TOTAL inventory for {sku_id}. "
                      f"Needed {total_qty_needed}, but only found "
                      f"{total_qty_needed - remaining_qty_to_find}. Route failed.")
                return None, current_planning_inventory 
        
        pickup_stops: Dict[int, List[Dict]] = {}
        for wh_node, sku_ops in temp_pickup_stops.items():
            pickup_stops[wh_node] = list(sku_ops.values())

        delivery_stops: Dict[int, List[Dict]] = {} 
        for order_id in vehicle_orders:
            order_node = order_nodes[order_id] 
            if order_node not in delivery_stops:
                delivery_stops[order_node] = []
            
            requirements = env.get_order_requirements(order_id)
            for sku, qty in requirements.items():
                delivery_op = {"order_id": order_id, "sku_id": sku, "quantity": qty}
                delivery_stops[order_node].append(delivery_op)

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
                print(f"Warning: Cannot find path to pickup node from {current_node}. Route failed.")
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
                print(f"Warning: Cannot find path to delivery node from {current_node}. Route failed.")
                return None, current_planning_inventory 
                
            if len(shortest_path_to_node) > 1:
                for node_id in shortest_path_to_node[1:-1]:
                    steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
            steps.append({"node_id": closest_node, "pickups": [], "deliveries": delivery_stops.get(closest_node, []), "unloads": []})
            current_node = closest_node
            delivery_nodes_to_visit.remove(closest_node)
            
        path_home_list, path_home_cost = shortest_path(current_node, home_node)
        if not path_home_list:
            print(f"Warning: Cannot find path home from {current_node}. Route failed.")
            return None, current_planning_inventory 

        if len(path_home_list) > 1:
            for node_id in path_home_list[1:-1]:
                steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
        steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})
        
        is_valid, msg = env.validator.validate_route_steps(vehicle_id, steps)
        if is_valid:
            return steps, planning_inventory
        else:
            print(f"Validation FAILED for {vehicle_id}: {msg}. Returning None.")
            return None, current_planning_inventory 
    # --- END HELPER FUNCTION ---


    # --- MAIN LOOP: "High-Throughput Greedy Bin Packing" ---
    
    # Create the "master" inventory state
    planning_inventory = {
        wh_id: wh.inventory.copy() for wh_id, wh in env.warehouses.items()
    }

    # Sort vehicles from LARGEST to SMALLEST
    all_vehicles.sort(key=lambda v: (v.capacity_weight, v.capacity_volume), reverse=True)
    
    unassigned_order_set = set(order_ids)
    
    for vehicle in all_vehicles:
        if not unassigned_order_set:
            break # All orders assigned
        
        vehicle_orders: List[str] = []
        current_weight = 0.0
        current_volume = 0.0
        
        # ---
        # This is the "High-Throughput" packing loop.
        # We pack based *only* on weight and volume,
        # then let the route generator validate the inventory.
        # ---
        
        # Sort by cluster ID to get *some* geographic benefit
        orders_to_check = sorted(
            list(unassigned_order_set), 
            key=lambda oid: order_to_cluster_map.get(oid, -1)
        )
        
        for order_id in orders_to_check:
            
            order_weight = order_weights[order_id]
            order_volume = order_volumes[order_id]
            
            # Check 1: Weight/Volume
            if (current_weight + order_weight <= vehicle.capacity_weight and
                current_volume + order_volume <= vehicle.capacity_volume):
                
                # This order fits! Add it to the truck.
                vehicle_orders.append(order_id)
                current_weight += order_weight
                current_volume += order_volume
                unassigned_order_set.remove(order_id)
            else:
                # This order does not fit. Skip it.
                continue
        # --- End Packing Loop ---

        # If we packed any orders, generate the route
        if vehicle_orders:
            vehicle_id = vehicle.id
            home_node = env.get_vehicle_home_warehouse(vehicle_id)
            
            steps, new_planning_inventory = generate_route_for_orders(
                vehicle_id, home_node, vehicle_orders, planning_inventory
            )
            
            if steps:
                # SUCCESS! Commit the route and the new inventory state
                solution["routes"].append({"vehicle_id": vehicle_id, "steps": steps})
                planning_inventory = new_planning_inventory # Commit
            else:
                # FAILURE! Roll back.
                print(f"Main route gen failed for {vehicle_id}. Returning orders to pool.")
                unassigned_order_set.update(vehicle_orders)
    
    # --- END MAIN LOOP ---
    
    if unassigned_order_set:
        print(f"Warning: {len(unassigned_order_set)} orders were NOT assigned after fallback.")

    return solution


if __name__ == "__main__":
    print("Running solver locally...")
    env = LogisticsEnvironment()
        
    env.set_solver(solver)
    
    # To run with the dashboard:
    # print("Launching dashboard... (Comment this out for headless)")
    # env.launch_dashboard()
    
    # To run headlessly:
    print("Running headless...")
    results = env.run_headless("optimized_solver_run")
    print("\n--- Solver Results ---")
    print(results)
    
    # --- FIX: RESET ENVIRONMENT STATE ---
    print("\nResetting environment state for statistics run...")
    env.reset_all_state() 
    # --- END FIX ---

    # --- SECOND RUN (for stats) ---
    print("Running solver again for stats...")
    solution = solver(env) 
    stats = env.get_solution_statistics(solution)
    print("\n--- Solution Statistics (from fresh run) ---")
    print(stats)