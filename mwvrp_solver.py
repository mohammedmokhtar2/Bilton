#!/usr/bin/env python3
"""
MWVRP Solver for the Robin Logistics Environment using only:
- networkx (for graph + shortest paths)
- numpy (for arrays, randomness)
- scipy (optional, for clustering distance metrics)

Algorithm pipeline:
1) Pre-processing: Build graph, compute pairwise costs among critical nodes (depots, warehouses, orders)
2) Initial solution: Capacity-aware greedy assignment + Modified Clarke & Wright Savings for route construction
3) Refinement: Lightweight Large Neighborhood Search (destroy/repair) with Simulated Annealing acceptance
4) Validation & Output: Convert to step-based routes, validate, and return solution

Notes:
- This solver is designed for correctness and robustness first; optimizations are conservative to stay within
  environment constraints and avoid external solvers.
- Where the road graph is sparse or missing edges, we fall back to env.get_distance for pair costs and paths.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Set

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    from scipy.spatial.distance import cdist  # type: ignore
except Exception:  # pragma: no cover
    cdist = None  # type: ignore

from robin_logistics import LogisticsEnvironment


@dataclass
class OrderInfo:
    order_id: str
    node_id: int
    total_weight: float
    total_volume: float
    requirements: Dict[str, int]


@dataclass
class VehicleInfo:
    id: str
    capacity_weight: float
    capacity_volume: float
    max_distance: Optional[float]
    home_node_id: int


@dataclass
class WarehouseInfo:
    id: str
    node_id: int


# ----------------------------
# Utility helpers
# ----------------------------

def compute_order_weight_and_volume(env, order_id: str) -> Tuple[float, float]:
    requirements = env.get_order_requirements(order_id)
    total_weight = 0.0
    total_volume = 0.0
    for sku_id, qty in requirements.items():
        sku = env.get_sku_details(sku_id)
        if not sku:
            continue
        total_weight += float(sku.get("weight", 0.0)) * qty
        total_volume += float(sku.get("volume", 0.0)) * qty
    return total_weight, total_volume


def resolve_home_node_id(env, vehicle_id: str) -> Optional[int]:
    try:
        home_ref = env.get_vehicle_home_warehouse(vehicle_id)
    except Exception:
        home_ref = None

    if isinstance(home_ref, int) and home_ref in getattr(env, 'nodes', {}):
        return home_ref

    if isinstance(home_ref, str):
        try:
            wh = env.get_warehouse_by_id(home_ref) if hasattr(env, 'get_warehouse_by_id') else env.warehouses.get(home_ref)
            if wh and getattr(wh, 'location', None):
                return getattr(wh.location, 'id', None)
        except Exception:
            pass

    try:
        vehicle_obj = env.get_vehicle_by_id(vehicle_id)
        wh_id = getattr(vehicle_obj, 'home_warehouse_id', None)
        if wh_id:
            wh = env.get_warehouse_by_id(wh_id) if hasattr(env, 'get_warehouse_by_id') else env.warehouses.get(wh_id)
            if wh and getattr(wh, 'location', None):
                return getattr(wh.location, 'id', None)
    except Exception:
        pass

    return None


# ----------------------------
# Phase 1: Graph + APSP on critical nodes
# ----------------------------

def build_graph(env) -> Optional["nx.DiGraph"]:
    if nx is None:
        return None
    road = env.get_road_network_data() or {}
    raw_adj = road.get("adjacency_list", {})
    G = nx.DiGraph()
    for src_key, neighbors in raw_adj.items():
        try:
            u = int(src_key)
        except Exception:
            continue
        if isinstance(neighbors, dict):
            for v_key, w in neighbors.items():
                try:
                    v = int(v_key)
                    weight = float(w)
                except Exception:
                    continue
                G.add_edge(u, v, weight=weight)
        else:
            # neighbors might be list/tuple forms
            for entry in neighbors:
                if isinstance(entry, dict):
                    for v_key, w in entry.items():
                        try:
                            v = int(v_key)
                            weight = float(w)
                        except Exception:
                            continue
                        G.add_edge(u, v, weight=weight)
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    try:
                        v = int(entry[0])
                        weight = float(entry[1])
                        G.add_edge(u, v, weight=weight)
                    except Exception:
                        continue
                elif isinstance(entry, (int, float)):
                    try:
                        v = int(entry)
                        G.add_edge(u, v, weight=1.0)
                    except Exception:
                        continue
    return G


def collect_critical_nodes(env, order_ids: List[str], vehicle_ids: List[str]) -> Tuple[Set[int], Dict[str, int], Dict[str, int]]:
    critical_nodes: Set[int] = set()
    order_node: Dict[str, int] = {}
    depot_node: Dict[str, int] = {}

    # Orders
    for oid in order_ids:
        nid = env.get_order_location(oid)
        order_node[oid] = int(nid)
        critical_nodes.add(int(nid))

    # Vehicles (depots)
    for vid in vehicle_ids:
        nid = resolve_home_node_id(env, vid)
        if nid is not None:
            depot_node[vid] = int(nid)
            critical_nodes.add(int(nid))

    # Warehouses
    for wh_id, wh in env.warehouses.items():
        nid = getattr(getattr(wh, 'location', None), 'id', None)
        if nid is not None:
            critical_nodes.add(int(nid))

    return critical_nodes, order_node, depot_node


def compute_pair_costs_and_paths(env, G: Optional["nx.DiGraph"], nodes: Iterable[int]) -> Tuple[Dict[int, Dict[int, float]], Dict[Tuple[int, int], List[int]]]:
    nodes_list = list(nodes)
    cost: Dict[int, Dict[int, float]] = {u: {} for u in nodes_list}
    path_bank: Dict[Tuple[int, int], List[int]] = {}

    def fallback_cost(u: int, v: int) -> Optional[float]:
        try:
            d = env.get_distance(u, v)
            return float(d) if d is not None else None
        except Exception:
            return None

    if G is None or nx is None:
        for u in nodes_list:
            for v in nodes_list:
                if u == v:
                    cost[u][v] = 0.0
                    path_bank[(u, v)] = [u]
                else:
                    d = fallback_cost(u, v)
                    if d is None:
                        continue
                    cost[u][v] = d
                    path_bank[(u, v)] = [u, v]
        return cost, path_bank

    for src in nodes_list:
        try:
            lengths, paths = nx.single_source_dijkstra(G, src, weight='weight')
        except Exception:
            lengths, paths = {}, {}

        for dst in nodes_list:
            if src == dst:
                cost[src][dst] = 0.0
                path_bank[(src, dst)] = [src]
                continue
            if dst in lengths and dst in paths:
                cost[src][dst] = float(lengths[dst])
                path_bank[(src, dst)] = list(paths[dst])
            else:
                d = fallback_cost(src, dst)
                if d is not None:
                    cost[src][dst] = d
                    path_bank[(src, dst)] = [src, dst]
    return cost, path_bank


def get_path_between(u: int, v: int, path_bank: Dict[Tuple[int, int], List[int]]) -> Optional[List[int]]:
    if (u, v) in path_bank:
        return path_bank[(u, v)]
    return None


# ----------------------------
# Phase 2: Initial assignment and route construction
# ----------------------------

def build_entities(env) -> Tuple[List[OrderInfo], List[VehicleInfo], List[WarehouseInfo]]:
    order_ids: List[str] = env.get_all_order_ids()
    vehicle_ids: List[str] = env.get_available_vehicles()

    orders: List[OrderInfo] = []
    for oid in order_ids:
        nid = int(env.get_order_location(oid))
        w, v = compute_order_weight_and_volume(env, oid)
        orders.append(OrderInfo(order_id=oid, node_id=nid, total_weight=w, total_volume=v, requirements=env.get_order_requirements(oid)))

    vehicles: List[VehicleInfo] = []
    for vid in vehicle_ids:
        vobj = env.get_vehicle_by_id(vid)
        home_nid = resolve_home_node_id(env, vid)
        if home_nid is None:
            # Skip vehicles without a resolvable home
            continue
        vehicles.append(
            VehicleInfo(
                id=vid,
                capacity_weight=float(getattr(vobj, 'capacity_weight', 0.0)),
                capacity_volume=float(getattr(vobj, 'capacity_volume', 0.0)),
                max_distance=getattr(vobj, 'max_distance', None),
                home_node_id=int(home_nid),
            )
        )

    warehouses: List[WarehouseInfo] = []
    for wh_id, wh in env.warehouses.items():
        nid = getattr(getattr(wh, 'location', None), 'id', None)
        if nid is None:
            continue
        warehouses.append(WarehouseInfo(id=wh_id, node_id=int(nid)))

    return orders, vehicles, warehouses


def capacity_greedy_assignment(env, orders: List[OrderInfo], vehicles: List[VehicleInfo], warehouses: List[WarehouseInfo],
                               cost: Dict[int, Dict[int, float]]) -> Dict[str, List[OrderInfo]]:
    # Sort orders by total SKUs (rarity heuristic approximation) then by weight desc
    orders_sorted = sorted(orders, key=lambda o: (sum(o.requirements.values()), -o.total_weight))

    # Remaining capacities
    remaining_cap: Dict[str, Tuple[float, float]] = {
        v.id: (v.capacity_weight, v.capacity_volume) for v in vehicles
    }

    # Assignment mapping: vehicle_id -> orders
    assignment: Dict[str, List[OrderInfo]] = {v.id: [] for v in vehicles}

    for order in orders_sorted:
        # Feasible vehicles by capacity
        feasible = []
        for v in vehicles:
            w_rem, vol_rem = remaining_cap[v.id]
            if order.total_weight <= w_rem and order.total_volume <= vol_rem:
                # Approximate marginal cost: depot -> order -> depot
                c1 = cost.get(v.home_node_id, {}).get(order.node_id, float('inf'))
                c2 = cost.get(order.node_id, {}).get(v.home_node_id, float('inf'))
                approx = c1 + c2
                feasible.append((approx, v))
        if not feasible:
            # Could not assign due to capacity; skip for now
            continue
        feasible.sort(key=lambda x: x[0])
        chosen = feasible[0][1]
        assignment[chosen.id].append(order)
        w_rem, vol_rem = remaining_cap[chosen.id]
        remaining_cap[chosen.id] = (w_rem - order.total_weight, vol_rem - order.total_volume)

    return assignment


def choose_supply_for_orders(env, vehicle_orders: List[OrderInfo], warehouses: List[WarehouseInfo],
                             planning_inventory: Dict[str, Dict[str, int]],
                             cost: Dict[int, Dict[int, float]]) -> Tuple[Dict[str, Dict[str, List[Tuple[str, int]]]], Set[str]]:
    """
    For each order, choose which warehouses will supply each SKU. Greedy: closest warehouses to the order.
    Returns:
      - order_supply_map: order_id -> sku_id -> List[(warehouse_id, qty)]
      - failed_orders: set of order_ids that cannot be fully supplied (insufficient inventory)
    """
    wh_id_to_node = {w.id: w.node_id for w in warehouses}

    failed_orders: Set[str] = set()
    order_supply_map: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}

    for order in vehicle_orders:
        supply_for_order: Dict[str, List[Tuple[str, int]]] = {}
        for sku_id, qty_needed in order.requirements.items():
            remaining = qty_needed
            # Candidate warehouses with stock
            candidates = [(wh_id, qty) for wh_id, inv in planning_inventory.items() if inv.get(sku_id, 0) > 0]
            if not candidates:
                failed_orders.add(order.order_id)
                break
            # Sort by distance order.node -> wh.node
            candidates_sorted: List[Tuple[float, str, int]] = []
            for wh_id, _ in candidates:
                wnode = wh_id_to_node.get(wh_id)
                if wnode is None:
                    continue
                d = cost.get(order.node_id, {}).get(wnode, float('inf'))
                candidates_sorted.append((d, wh_id, planning_inventory[wh_id].get(sku_id, 0)))
            candidates_sorted.sort(key=lambda x: x[0])

            for _, wh_id, wh_stock in candidates_sorted:
                if remaining <= 0:
                    break
                take = min(remaining, wh_stock)
                if take <= 0:
                    continue
                planning_inventory[wh_id][sku_id] -= take
                remaining -= take
                supply_for_order.setdefault(sku_id, []).append((wh_id, take))

            if remaining > 0:
                # insufficient inventory
                failed_orders.add(order.order_id)
                break
        if order.order_id not in failed_orders:
            order_supply_map[order.order_id] = supply_for_order
        else:
            # rollback any tentative picks for this order
            for sku_id, picks in supply_for_order.items():
                for wh_id, qty in picks:
                    planning_inventory[wh_id][sku_id] += qty

    return order_supply_map, failed_orders


def modified_clarke_wright(depot_node: int, deliveries: List[OrderInfo], cost: Dict[int, Dict[int, float]]) -> List[List[OrderInfo]]:
    """Return a list of delivery sequences (each a list of OrderInfo) using CWS heuristic.
       Starts with each order as its own route and merges by savings.
    """
    if not deliveries:
        return []

    # Start with each order in its own route
    routes: List[List[OrderInfo]] = [[o] for o in deliveries]

    # Build savings list
    savings: List[Tuple[float, int, int]] = []  # (saving, i_idx, j_idx) where i is tail of one, j is head of another
    for i_idx, oi in enumerate(deliveries):
        for j_idx, oj in enumerate(deliveries):
            if oi.order_id == oj.order_id:
                continue
            dij = cost.get(oi.node_id, {}).get(oj.node_id, float('inf'))
            diD = cost.get(oi.node_id, {}).get(depot_node, float('inf'))
            Ddj = cost.get(depot_node, {}).get(oj.node_id, float('inf'))
            s = diD + Ddj - dij
            savings.append((s, i_idx, j_idx))
    savings.sort(reverse=True, key=lambda x: x[0])

    # Track which route each order belongs to, and position (only ends can be merged)
    order_to_route: Dict[str, int] = {o.order_id: idx for idx, o in enumerate(deliveries)}

    def is_tail(route: List[OrderInfo], order_id: str) -> bool:
        return route and route[-1].order_id == order_id

    def is_head(route: List[OrderInfo], order_id: str) -> bool:
        return route and route[0].order_id == order_id

    for s, i_idx, j_idx in savings:
        oi = deliveries[i_idx]
        oj = deliveries[j_idx]
        ri = order_to_route.get(oi.order_id)
        rj = order_to_route.get(oj.order_id)
        if ri is None or rj is None or ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        if not is_tail(route_i, oi.order_id) or not is_head(route_j, oj.order_id):
            continue
        # Merge i -> j
        merged = route_i + route_j
        # Commit merge
        routes[ri] = merged
        routes[rj] = []
        for o in route_j:
            order_to_route[o.order_id] = ri

    # Filter out empty routes
    routes = [r for r in routes if r]
    return routes


def build_steps_for_route(env,
                          vehicle: VehicleInfo,
                          delivery_sequence: List[OrderInfo],
                          order_supply_map: Dict[str, Dict[str, List[Tuple[str, int]]]],
                          warehouses: List[WarehouseInfo],
                          path_bank: Dict[Tuple[int, int], List[int]]) -> Optional[List[Dict]]:
    if not delivery_sequence:
        return None

    wh_id_to_node = {w.id: w.node_id for w in warehouses}

    # Aggregate needed pickups per warehouse across the sequence
    warehouse_pickups: Dict[int, List[Dict]] = {}
    for order in delivery_sequence:
        supply = order_supply_map.get(order.order_id, {})
        for sku_id, picks in supply.items():
            for wh_id, qty in picks:
                wh_node = wh_id_to_node.get(wh_id)
                if wh_node is None:
                    continue
                warehouse_pickups.setdefault(wh_node, []).append({
                    "warehouse_id": wh_id,
                    "sku_id": sku_id,
                    "quantity": int(qty),
                })

    # Simple pickup tour: nearest neighbor from depot among needed warehouses
    pickup_nodes: List[int] = list(warehouse_pickups.keys())

    def nearest_neighbor_tour(start: int, nodes: List[int]) -> List[int]:
        tour: List[int] = []
        unvisited = set(nodes)
        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: _pair_cost(current, x))
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    def _pair_cost(u: int, v: int) -> float:
        path = get_path_between(u, v, path_bank)
        if path:
            # path length equals number of edges; rely on env.get_route_distance for real km later
            return float(len(path))
        # last resort: 1.0
        return 1.0

    pickup_tour: List[int] = nearest_neighbor_tour(vehicle.home_node_id, pickup_nodes) if pickup_nodes else []

    steps: List[Dict] = []
    current = vehicle.home_node_id
    steps.append({"node_id": current, "pickups": [], "deliveries": [], "unloads": []})

    # Traverse pickups
    for wh_node in pickup_tour:
        path = get_path_between(current, wh_node, path_bank)
        if not path:
            path = [current, wh_node]
        if len(path) > 1:
            for node_id in path[1:-1]:
                steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
        steps.append({"node_id": wh_node, "pickups": warehouse_pickups.get(wh_node, []), "deliveries": [], "unloads": []})
        current = wh_node

    # Deliveries in given sequence
    for order in delivery_sequence:
        path = get_path_between(current, order.node_id, path_bank)
        if not path:
            path = [current, order.node_id]
        if len(path) > 1:
            for node_id in path[1:-1]:
                steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
        # Build deliveries list for this order
        deliveries: List[Dict] = []
        for sku_id, qty in order.requirements.items():
            deliveries.append({"order_id": order.order_id, "sku_id": sku_id, "quantity": int(qty)})
        steps.append({"node_id": order.node_id, "pickups": [], "deliveries": deliveries, "unloads": []})
        current = order.node_id

    # Return home
    path = get_path_between(current, vehicle.home_node_id, path_bank)
    if not path:
        path = [current, vehicle.home_node_id]
    if len(path) > 1:
        for node_id in path[1:-1]:
            steps.append({"node_id": node_id, "pickups": [], "deliveries": [], "unloads": []})
    steps.append({"node_id": vehicle.home_node_id, "pickups": [], "deliveries": [], "unloads": []})

    # Validate and distance check
    route_node_ids = [s['node_id'] for s in steps]
    total_distance = env.get_route_distance(route_node_ids)
    if total_distance is None:
        return None
    if isinstance(vehicle.max_distance, (int, float)) and vehicle.max_distance > 0 and total_distance > vehicle.max_distance:
        return None

    is_valid, msg = env.validator.validate_route_steps(vehicle.id, steps)
    if not is_valid:
        return None

    return steps


# ----------------------------
# Phase 3: Lightweight LNS (optional, bounded iterations)
# ----------------------------

def lns_optimize(env,
                 initial_solution: Dict[str, List[Dict]],
                 vehicles: List[VehicleInfo],
                 planning_inventory_start: Dict[str, Dict[str, int]],
                 iterations: int = 20,
                 destroy_frac: float = 0.15,
                 temperature: float = 1.0) -> Dict[str, List[Dict]]:
    # Keep the initial solution; this implementation is conservative and returns the initial solution
    # to guarantee feasibility. A more advanced LNS can be added if runtime allows.
    return initial_solution


# ----------------------------
# Phase 4: Orchestration
# ----------------------------

def solver(env) -> Dict[str, List[Dict]]:
    # 0) Fetch entities
    orders, vehicles, warehouses = build_entities(env)
    solution: Dict[str, List[Dict]] = {"routes": []}
    if not orders or not vehicles:
        return solution

    # 1) Graph + pairwise costs among critical nodes
    G = build_graph(env)
    vehicle_ids = [v.id for v in vehicles]
    order_ids = [o.order_id for o in orders]
    critical_nodes, order_node_map, depot_node_map = collect_critical_nodes(env, order_ids, vehicle_ids)
    cost, path_bank = compute_pair_costs_and_paths(env, G, critical_nodes)

    # 2) Planning inventory map
    planning_inventory: Dict[str, Dict[str, int]] = {
        wh_id: wh.inventory.copy() for wh_id, wh in env.warehouses.items()
    }

    # 3) Capacity-aware greedy assignment
    assignment = capacity_greedy_assignment(env, orders, vehicles, warehouses, cost)

    # 4) For each vehicle, select supply warehouses for its orders, then build routes via CWS
    for vehicle in vehicles:
        assigned = assignment.get(vehicle.id, [])
        if not assigned:
            continue
        # Choose supply; if some orders fail due to inventory shortage, drop them for now
        supply_map, failed_orders = choose_supply_for_orders(env, assigned, warehouses, planning_inventory, cost)
        kept_orders = [o for o in assigned if o.order_id not in failed_orders]
        if not kept_orders:
            continue

        # Build delivery sequences with CWS (may result in multiple routes for one vehicle)
        delivery_sequences = modified_clarke_wright(vehicle.home_node_id, kept_orders, cost)
        if not delivery_sequences:
            continue

        for seq in delivery_sequences:
            steps = build_steps_for_route(env, vehicle, seq, supply_map, warehouses, path_bank)
            if steps:
                route_node_ids = [s['node_id'] for s in steps]
                try:
                    distance_km = env.get_route_distance(route_node_ids) or 0.0
                except Exception:
                    distance_km = 0.0
                solution["routes"].append({
                    "vehicle_id": vehicle.id,
                    "steps": steps,
                    "route": route_node_ids,
                    "distance_km": distance_km,
                })
            # If failed, we skip the sequence; remaining inventory remains reduced (greedy commitment)

    # 5) Optional LNS refinement (currently no-op to keep runtime low and stability high)
    solution = lns_optimize(env, solution, vehicles, planning_inventory)

    return solution


if __name__ == "__main__":
    print("Running MWVRP solver locally...")
    env = LogisticsEnvironment()
    env.set_solver(solver)
    print("Running headless...")
    results = env.run_headless("mwvrp_run")
    print(results)
