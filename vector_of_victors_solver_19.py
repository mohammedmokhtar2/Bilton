#!/usr/bin/env python3
"""
Advanced ALNS Solver for the Beltone AI Hackathon.

This solver implements an Adaptive Large Neighborhood Search (ALNS) metaheuristic.
This version incorporates all bug fixes related to API mismatches.

1.  Initial Solution: Generated using K-Means clustering and a greedy
    inventory-aware bin-packing heuristic.
2.  ALNS Loop:
    - Destroy: Randomly removes routes from the solution.
    - Repair: Re-inserts the removed orders using a greedy insertion heuristic
      that finds the cheapest feasible insertion point.
    - Acceptance: Uses a Simulated Annealing criterion to accept new solutions,
      allowing it to escape local optima.
3.  Feasibility: All insertions and routes are validated against inventory,
    vehicle capacity (weight/volume), and max_distance constraints.
"""

import copy
import heapq
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Set

# Optional scientific deps: fall back gracefully if unavailable
try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from scipy.cluster.vq import kmeans, vq  # type: ignore
except ImportError:  # pragma: no cover
    kmeans = None  # type: ignore
    vq = None  # type: ignore

from robin_logistics import LogisticsEnvironment

# --- Configuration ---
# Set to False for submission
DEBUG = False

# ALNS Configuration
ALNS_ITERATIONS = 500  # Number of iterations to run
ALNS_INITIAL_TEMPERATURE = 1000
ALNS_COOLING_RATE = 0.995
ALNS_REMOVAL_PERCENTAGE = 0.2  # Percentage of orders to remove

# K-Means Clustering Configuration
# Use clustering if numpy/scipy are available and orders > K
ENABLE_CLUSTERING = (np is not None) and (kmeans is not None)
# Try to match number of clusters to number of vehicles
K_CLUSTERS_FALLBACK = 20
K_MEANS_SEED = 42

# --- Data Models (for easier state management) ---

class Route:
    """Represents a single vehicle route."""
    def __init__(self, vehicle_id: str, home_node: int):
        self.vehicle_id: str = vehicle_id
        self.home_node: int = home_node
        # stops are (node_id, "pickup"|"delivery", "order_id"|"sku_id", quantity)
        # Simplified: We will use a list of stops.
        # Let's use a more structured stop:
        # {'node': int, 'type': 'pickup'|'delivery', 'order_id'?: str, 'items': Dict[sku, qty]}
        self.stops: List[Dict] = [{'node': home_node, 'type': 'home_start', 'items': {}}]
        self.current_weight: float = 0.0
        self.current_volume: float = 0.0
        self.current_distance: float = 0.0
        self.orders_served: Set[str] = set()

    def add_stop(self, stop: Dict, env: LogisticsEnvironment, vehicle_cap: Tuple[float, float]):
        """Adds a stop and updates route state. Assumes feasibility is pre-checked."""
        last_node = self.stops[-1]['node']
        new_node = stop['node']
        
        # --- FIX: Use g.shortest_path_length ---
        distance_to_stop = env.g.shortest_path_length(source=last_node, target=new_node, weight='length')
        self.current_distance += distance_to_stop

        if stop['type'] == 'pickup':
            for sku, qty in stop['items'].items():
                sku_details = env.get_sku_details(sku)
                if sku_details:
                    self.current_weight += sku_details['weight'] * qty
                    self.current_volume += sku_details['volume'] * qty
        
        elif stop['type'] == 'delivery':
            order_id = stop['order_id']
            self.orders_served.add(order_id)
            # We get reqs from cache, but here we just need to remove weight/volume
            for sku, qty in stop['items'].items():
                sku_details = env.get_sku_details(sku)
                if sku_details:
                    # Delivery *removes* weight/volume
                    self.current_weight -= sku_details['weight'] * qty
                    self.current_volume -= sku_details['volume'] * qty
        
        self.stops.append(stop)

    def finalize_route(self, env: LogisticsEnvironment):
        """Adds the final return-to-home stop."""
        if not self.is_finalized():
            last_node = self.stops[-1]['node']
            # --- FIX: Use g.shortest_path_length ---
            distance_to_home = env.g.shortest_path_length(source=last_node, target=self.home_node, weight='length')
            self.current_distance += distance_to_home
            self.stops.append({'node': self.home_node, 'type': 'home_end', 'items': {}})

    def is_finalized(self) -> bool:
        return self.stops and self.stops[-1]['type'] == 'home_end'

    def to_solution_steps(self, env: LogisticsEnvironment) -> List[Dict]:
        """Converts internal stops to the format required by the environment."""
        solution_steps = []
        
        # Ensure route is finalized before converting
        if not self.is_finalized():
            if DEBUG:
                print(f"Warning: Route {self.vehicle_id} not finalized before conversion.")
            # self.finalize_route(env) # This can be problematic if called mid-build

        # Group operations by node
        node_operations = defaultdict(lambda: {
            'node_id': 0, 'pickups': [], 'deliveries': [], 'unloads': []
        })

        # We skip home_start and home_end, as they are implied
        for stop in self.stops:
            node_id = stop['node']
            op_node = node_operations[node_id]
            op_node['node_id'] = node_id

            if stop['type'] == 'pickup':
                wh_id = stop['warehouse_id']
                for sku, qty in stop['items'].items():
                    op_node['pickups'].append({
                        'warehouse_id': wh_id,
                        'sku_id': sku,
                        'quantity': qty
                    })
            elif stop['type'] == 'delivery':
                op_node['deliveries'].append({
                    'order_id': stop['order_id'],
                    'items': stop['items']
                })
        
        # Need to determine the *travel order* of these nodes
        # The simple self.stops list already defines this order
        visited_nodes = set()
        travel_order = []
        for stop in self.stops[1:]: # Skip home_start
            if stop['type'] == 'home_end':
                continue
            if stop['node'] not in visited_nodes or stop['type'] == 'delivery':
                 # Allow multiple visits to same node if it's for delivery
                 # Or if it's a different stop type
                travel_order.append(stop['node'])
                visited_nodes.add(stop['node'])
        
        # Deduplicate while preserving order
        ordered_nodes = list(dict.fromkeys(travel_order))

        for node_id in ordered_nodes:
            if node_id in node_operations:
                solution_steps.append(node_operations[node_id])

        return solution_steps

class Solution:
    """Represents a complete solution state."""
    def __init__(self, env: LogisticsEnvironment):
        self.env: LogisticsEnvironment = env
        self.routes: Dict[str, Route] = {}
        self.unassigned_orders: Set[str] = set(env.get_all_order_ids())
        self.cost: float = float('inf')
        self.cost_calculated: bool = False

    def copy(self):
        """Creates a deep copy of the solution state."""
        new_sol = Solution(self.env)
        new_sol.routes = copy.deepcopy(self.routes)
        new_sol.unassigned_orders = copy.deepcopy(self.unassigned_orders)
        new_sol.cost = self.cost
        new_sol.cost_calculated = self.cost_calculated
        return new_sol

    def calculate_cost(self, force: bool = False) -> float:
        """Calculates the total cost of the solution using env validation."""
        if self.cost_calculated and not force:
            return self.cost

        # We must finalize all routes before calculating cost
        for route in self.routes.values():
            if not route.is_finalized():
                route.finalize_route(self.env)

        solution_dict = self.to_dict()
        
        # Use the environment to get the official cost
        try:
            # Note: This validation can be slow.
            # For pure ALNS, we might use a faster, estimated cost delta.
            # But for hackathon scoring, using the official calculator is safest.
            is_valid, message = self.env.validate_solution_complete(solution_dict)
            if not is_valid:
                if DEBUG:
                    print(f"Warning: ALNS generated an invalid solution: {message}")
                # Penalize invalid solutions heavily
                self.cost = float('inf')
            else:
                cost_breakdown = self.env.metrics_calculator.calculate_cost_breakdown(solution_dict)
                self.cost = cost_breakdown['total_cost']

        except Exception as e:
            if DEBUG:
                print(f"Error during cost calculation: {e}")
            self.cost = float('inf')
        
        self.cost_calculated = True
        return self.cost

    def to_dict(self) -> Dict[str, List[Dict]]:
        """Converts the Solution object to the submission dictionary format."""
        solution_dict = {}
        for vehicle_id, route in self.routes.items():
            if route.stops and len(route.stops) > 2: # More than just home_start/home_end
                steps = route.to_solution_steps(self.env)
                if steps: # Only add if there are actual operations
                    solution_dict[vehicle_id] = steps
        return solution_dict

    def add_route(self, route: Route):
        """Adds a new route to the solution."""
        self.routes[route.vehicle_id] = route
        self.unassigned_orders.difference_update(route.orders_served)
        self.cost_calculated = False # Cost is now stale

    def remove_order(self, order_id: str) -> Optional[Route]:
        """
        (This function is complex and not used by the current random_route_removal,
         but is kept for future destroy operator development)
        Removes an order from its route. Returns the modified route.
        """
        for route in self.routes.values():
            if order_id in route.orders_served:
                # This is a simple removal. A real implementation would
                # need to rebuild the route state, which is complex.
                # For this ALNS, we will rebuild the *entire route*
                # or just remove the stops associated with this order.
                
                # Simple version: find delivery, remove it and its pickups
                # This is complex. A simpler "destroy" is to remove the *entire route*.
                # Let's try that.
                
                # No, let's stick to order removal.
                # Find all stops related to this order
                delivery_stop_idx = -1
                for i, stop in enumerate(route.stops):
                    if stop.get('order_id') == order_id:
                        delivery_stop_idx = i
                        break
                
                if delivery_stop_idx != -1:
                    # Found it. Now, this is tricky. We need to remove pickups.
                    # This implies a full route rebuild.
                    
                    # For ALNS, a simpler way is often to just remove the order
                    # from the route's set and add it to unassigned.
                    # The route's *cost* will be wrong until recalculation,
                    # but the *state* (unassigned orders) is correct.
                    
                    # Let's rebuild the route from scratch without this order.
                    orders_on_route = list(route.orders_served)
                    orders_on_route.remove(order_id)
                    
                    vehicle_id = route.vehicle_id
                    
                    # This is too complex for a destroy op.
                    # Let's just remove the order from the list and let
                    # the cost function handle the (now invalid) route.
                    # A better way: just add to unassigned and let repair fix it.
                    
                    # Re-thinking: The "destroy" op just selects orders to remove.
                    # The "repair" op builds new routes or inserts into existing.
                    # This implies our destroy op should be stateful.
                    
                    route.orders_served.remove(order_id)
                    self.unassigned_orders.add(order_id)
                    self.cost_calculated = False
                    
                    # Now, we must remove the actual stops
                    route.stops = [s for s in route.stops if s.get('order_id') != order_id]
                    
                    # We also need to remove the *pickups* for this order.
                    # This is the hard part.
                    # Let's simplify: "destroy" removes N *random orders*
                    # from the *unassigned list* of a *copied solution*.
                    # No, that's wrong.
                    
                    # Correct ALNS "destroy":
                    # 1. Pick an order `order_id` to remove.
                    # 2. Find its route `r`.
                    # 3. Remove `order_id` from `r.orders_served`.
                    # 4. Add `order_id` to `sol.unassigned_orders`.
                    # 5. Re-calculate `r`'s structure (stops, cost, load)
                    #    * This is the hard part. *
                    #
                    # Let's use a simpler destroy:
                    # 1. Pick an order `order_id` to remove.
                    # 2. Find its route `r`.
                    # 3. Add *all* orders from `r` to `sol.unassigned_orders`.
                    # 4. Delete `r` from `sol.routes`.
                    # 5. Return the list of removed orders.
                    
                    removed_orders = list(route.orders_served)
                    self.unassigned_orders.update(removed_orders)
                    del self.routes[route.vehicle_id]
                    self.cost_calculated = False
                    return removed_orders
        
        return None # Order not found


# --- Main ALNS Solver Class ---

class ALNSSolver:
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.all_vehicles = env.get_all_vehicles()
        self.all_order_ids = env.get_all_order_ids()
        
        # --- FIX for get_all_warehouses and node_id AttributeErrors ---
        # Discover all warehouse IDs and their node IDs from the vehicles
        self.all_warehouse_ids = set()
        # This will map wh_id (str) -> node_id (int)
        self.warehouse_nodes: Dict[str, int] = {} 
        
        for v in self.all_vehicles:
            wh_id = v.home_warehouse_id
            if wh_id not in self.all_warehouse_ids:
                self.all_warehouse_ids.add(wh_id)
                # This API call *directly* gets the node ID (int)
                # for the vehicle's home warehouse.
                try:
                    home_node_id = self.env.get_vehicle_home_warehouse(v.id)
                    if home_node_id is not None:
                        self.warehouse_nodes[wh_id] = home_node_id
                    else:
                        # Fallback if API fails
                        print(f"Warning: Could not find home node for vehicle {v.id}")
                except Exception as e:
                     print(f"Error getting home warehouse for {v.id}: {e}")

        # Convert set to sorted list for stable iteration later
        self.all_warehouse_ids = sorted(list(self.all_warehouse_ids))
        
        if not self.warehouse_nodes:
             print("CRITICAL: Warehouse node map is empty. Check environment setup.")
        # --- END FIX ---

        self.order_locations = self._get_order_locations()
        # --- FIX for KeyError: 'skus' ---
        # This cache is now built correctly
        self.order_reqs_cache = self._get_all_order_reqs() 
        
        self.current_solution = Solution(env)
        self.best_solution = Solution(env)
        
        self.temperature = ALNS_INITIAL_TEMPERATURE
        
        # Track initial inventory (uses all_warehouse_ids from FIX above)
        self.available_inventory = self._get_initial_inventory()
        
        # Order clustering
        self.order_clusters = self._cluster_orders()

    def _get_order_locations(self) -> Dict[str, int]:
        """Cache all order locations."""
        return {
            order_id: self.env.get_order_location(order_id)
            for order_id in self.all_order_ids
        }

    def _get_all_order_reqs(self) -> Dict[str, Dict]:
        """Cache all order requirements (weight, volume, skus)."""
        # --- REPLACED FUNCTION (FIX FOR KEYERROR) ---
        # This function was the source of the KeyError.
        # It now correctly processes the direct SKU dict from get_order_requirements
        # and builds a new, clean cache object.
        reqs_cache = {}
        for order_id in self.all_order_ids:
            # 1. Get the SKU dictionary directly
            sku_dict = self.env.get_order_requirements(order_id)
            
            # Handle cases where the API might return None or empty
            if not sku_dict:
                sku_dict = {}

            total_weight = 0.0
            total_volume = 0.0
            
            # 2. Iterate over the SKU dictionary items directly
            for sku, qty in sku_dict.items():
                sku_details = self.env.get_sku_details(sku)
                if sku_details:
                    total_weight += sku_details['weight'] * qty
                    total_volume += sku_details['volume'] * qty
            
            # 3. Store in a new, structured dictionary
            reqs_cache[order_id] = {
                'skus': sku_dict, # Store the original SKUs under the 'skus' key
                'total_weight': total_weight,
                'total_volume': total_volume
            }
        return reqs_cache
        # --- END REPLACED FUNCTION ---

    def _get_initial_inventory(self) -> Dict[str, Dict[str, int]]:
        """Cache the starting inventory for all warehouses."""
        # --- FIX: Iterate over self.all_warehouse_ids ---
        return {
            wh_id: self.env.get_warehouse_inventory(wh_id)
            for wh_id in self.all_warehouse_ids
        }
        # --- END FIX ---

    def _cluster_orders(self) -> Dict[str, int]:
        """Group orders into geographic clusters using K-Means."""
        if not ENABLE_CLUSTERING or len(self.all_order_ids) < K_CLUSTERS_FALLBACK:
            if DEBUG:
                print("Clustering disabled or too few orders.")
            return {order_id: 0 for order_id in self.all_order_ids}

        try:
            order_nodes = [self.order_locations[oid] for oid in self.all_order_ids]
            
            # Get (lat, lon) for each node
            coords = np.array([
                self.env.g.nodes[node_id]['pos'] for node_id in order_nodes
            ])
            
            # Determine K
            num_vehicles = len(self.all_vehicles)
            n_clusters = min(num_vehicles, len(self.all_order_ids))
            if n_clusters == 0:
                 return {order_id: 0 for order_id in self.all_order_ids}
            
            if DEBUG:
                print(f"Clustering {len(coords)} orders into {n_clusters} zones...")

            # Run K-Means
            np.random.seed(K_MEANS_SEED)
            centroids, _ = kmeans(coords, n_clusters)
            # Assign each order to a cluster
            cluster_indices, _ = vq(coords, centroids)
            
            return dict(zip(self.all_order_ids, cluster_indices))

        except Exception as e:
            if DEBUG:
                print(f"Clustering failed: {e}. Falling back to single cluster.")
            return {order_id: 0 for order_id in self.all_order_ids}

    def solve(self) -> Solution:
        """Main ALNS solver loop."""
        
        print("Generating initial solution using greedy cluster-packing...")
        self.current_solution = self.generate_initial_solution()
        self.current_solution.calculate_cost()
        self.best_solution = self.current_solution.copy()
        
        print(f"Initial Cost: {self.best_solution.cost}")
        
        for i in range(ALNS_ITERATIONS):
            new_solution = self.current_solution.copy()
            
            # 1. Destroy
            removed_orders, modified_routes = self.random_route_removal(new_solution, num_to_remove=2)

            if not removed_orders:
                # No routes to remove, solution is empty or stuck
                continue

            # 2. Repair
            # Use greedy insertion for the removed orders
            self.greedy_insertion_repair(new_solution, removed_orders, self.available_inventory)
            
            # 3. Calculate Cost
            new_solution.calculate_cost(force=True)
            
            # 4. Accept
            cost_delta = new_solution.cost - self.current_solution.cost
            
            if cost_delta < 0:
                # Better solution found
                self.current_solution = new_solution
                if new_solution.cost < self.best_solution.cost:
                    self.best_solution = new_solution
                    if DEBUG:
                        print(f"  > Iter {i}: New Best Cost: {self.best_solution.cost}")
            
            elif self.temperature > 0.1 and math.exp(-cost_delta / self.temperature) > random.random():
                # Accept worse solution (Simulated Annealing)
                self.current_solution = new_solution
                if DEBUG:
                    print(f"  > Iter {i}: Accepted worse cost: {self.current_solution.cost} (Temp: {self.temperature:.2f})")
            
            # Update temperature
            self.temperature *= ALNS_COOLING_RATE

            if i % 100 == 0 and DEBUG:
                print(f"Iter {i}: Best: {self.best_solution.cost}, Current: {self.current_solution.cost}, Temp: {self.temperature:.2f}")

        print(f"\nALNS finished. Best cost found: {self.best_solution.cost}")
        
        # Final fallback for any remaining unassigned orders
        if self.best_solution.unassigned_orders:
             print(f"Running final fallback insertion for {len(self.best_solution.unassigned_orders)} orders...")
             self.greedy_insertion_repair(
                 self.best_solution, 
                 list(self.best_solution.unassigned_orders), 
                 self.available_inventory,
                 allow_new_routes=True
             )
             self.best_solution.calculate_cost(force=True)
             print(f"Final cost after fallback: {self.best_solution.cost}")
             if self.best_solution.unassigned_orders:
                 print(f"Warning: {len(self.best_solution.unassigned_orders)} orders still unassigned.")

        return self.best_solution

    def generate_initial_solution(self) -> Solution:
        """
        Generates the initial solution using the cluster-based greedy packing
        heuristic from solver_13.
        """
        solution = Solution(self.env)
        
        # Use a deep copy of inventory for this simulation
        inventory = copy.deepcopy(self.available_inventory)
        
        # Sort vehicles: smallest, cheapest first
        sorted_vehicles = sorted(
            self.all_vehicles,
            key=lambda v: (v.capacity_weight, v.fixed_cost, v.cost_per_km)
        )
        
        # Group orders by cluster
        orders_by_cluster = defaultdict(list)
        for order_id, cluster_id in self.order_clusters.items():
            orders_by_cluster[cluster_id].append(order_id)
            
        unassigned_order_set = set(self.all_order_ids)
        
        for vehicle in sorted_vehicles:
            if not unassigned_order_set:
                break # All orders assigned
            
            # Try to pack this vehicle
            
            # Find the best cluster to serve for this vehicle
            # This is complex. Let's simplify:
            # Iterate through clusters and try to fill the vehicle.
            
            best_cluster_id = -1
            if self.order_clusters:
                # Find cluster with most unassigned orders
                try:
                    best_cluster_id = max(
                        orders_by_cluster.keys(),
                        key=lambda cid: len([
                            oid for oid in orders_by_cluster[cid] if oid in unassigned_order_set
                        ])
                    )
                except ValueError: # No orders left to assign
                    break
            
            orders_to_pack = []
            if best_cluster_id != -1:
                # Prioritize orders from the best cluster
                orders_to_pack = [
                    oid for oid in orders_by_cluster[best_cluster_id] 
                    if oid in unassigned_order_set
                ]
            
            # Add remaining orders as fillers
            orders_to_pack.extend([
                oid for oid in unassigned_order_set if oid not in orders_to_pack
            ])
            
            # --- Start Packing Logic ---
            current_weight = 0.0
            current_volume = 0.0
            orders_for_this_route = []
            
            for order_id in orders_to_pack:
                # Check if already assigned by another vehicle in this loop
                if order_id not in unassigned_order_set:
                    continue
                    
                order_reqs = self.order_reqs_cache[order_id]
                order_w = order_reqs['total_weight']
                order_v = order_reqs['total_volume']
                
                # Check capacity
                if (current_weight + order_w <= vehicle.capacity_weight and
                    current_volume + order_v <= vehicle.capacity_volume):
                    
                    # Check inventory
                    has_inventory, _ = self._find_pickups_for_orders(
                        [order_id], inventory
                    )
                    
                    if has_inventory:
                        current_weight += order_w
                        current_volume += order_v
                        orders_for_this_route.append(order_id)

            # --- End Packing Logic ---

            if orders_for_this_route:
                # Build the route for this vehicle
                route, route_inventory = self._build_route_for_orders(
                    vehicle, orders_for_this_route, inventory
                )
                
                if route:
                    # Check distance constraint
                    route.finalize_route(self.env)
                    if route.current_distance <= vehicle.max_distance:
                        solution.add_route(route)
                        unassigned_order_set.difference_update(orders_for_this_route)
                        # Commit inventory changes
                        inventory = route_inventory
                    else:
                        if DEBUG:
                            print(f"Initial route for {vehicle.id} failed max_distance: {route.current_distance} > {vehicle.max_distance}")
        
        solution.unassigned_orders = unassigned_order_set
        return solution

    def _find_pickups_for_orders(self, order_ids: List[str], current_inventory: Dict) -> Tuple[bool, Dict[str, Dict[str, int]]]:
        """
        Finds the best warehouses to pick up all items for a list of orders.
        Returns (success, {warehouse_id: {sku: qty}})
        """
        total_skus_needed = defaultdict(int)
        for order_id in order_ids:
            # --- FIX: Access the new 'skus' key from the cache ---
            for sku, qty in self.order_reqs_cache[order_id]['skus'].items():
                total_skus_needed[sku] += qty
        
        pickup_plan = defaultdict(lambda: defaultdict(int))
        
        for sku, needed_qty in total_skus_needed.items():
            found_qty = 0
            
            # Find warehouses that have this SKU
            # This is slow, better to use the cached inventory
            warehouses_with_sku = []
            for wh_id, inv in current_inventory.items():
                if inv.get(sku, 0) > 0:
                    warehouses_with_sku.append(wh_id)
            
            # Sort warehouses (e.g., by proximity, but for now, by ID)
            warehouses_with_sku.sort() 
            
            for wh_id in warehouses_with_sku:
                available = current_inventory[wh_id].get(sku, 0)
                take_qty = min(needed_qty - found_qty, available)
                
                if take_qty > 0:
                    pickup_plan[wh_id][sku] += take_qty
                    found_qty += take_qty
                
                if found_qty == needed_qty:
                    break
            
            if found_qty < needed_qty:
                # Not enough inventory in the *entire network*
                return False, {}
                
        return True, pickup_plan

    def _build_route_for_orders(self, vehicle: object, order_ids: List[str], 
                                current_inventory: Dict) -> Tuple[Optional[Route], Optional[Dict]]:
        """
        Builds a single feasible route (pickups + deliveries) for a set of orders.
        Applies "Pickups-First" TSP-like routing.
        """
        
        # 1. Find pickup plan
        has_inv, pickup_plan = self._find_pickups_for_orders(order_ids, current_inventory)
        if not has_inv:
            return None, None
            
        # 2. Create new inventory state *after* pickups
        new_inventory = copy.deepcopy(current_inventory)
        for wh_id, skus in pickup_plan.items():
            for sku, qty in skus.items():
                if sku in new_inventory[wh_id]:
                    new_inventory[wh_id][sku] -= qty
                
        # 3. Build route stops
        home_node = self.env.get_vehicle_home_warehouse(vehicle.id)
        if home_node is None: # Fallback if home warehouse not found
            home_node = self.warehouse_nodes[vehicle.home_warehouse_id]
            
        route = Route(vehicle.id, home_node)
        
        # 4. Create pickup and delivery stops
        pickup_stops = []
        for wh_id, items in pickup_plan.items():
            pickup_stops.append({
                'node': self.warehouse_nodes[wh_id],
                'type': 'pickup',
                'warehouse_id': wh_id,
                'items': items
            })
            
        delivery_stops = []
        for order_id in order_ids:
            reqs = self.order_reqs_cache[order_id]
            delivery_stops.append({
                'node': self.order_locations[order_id],
                'type': 'delivery',
                'order_id': order_id,
                # --- FIX: Access the new 'skus' key from the cache ---
                'items': reqs['skus']
            })
            
        # 5. Solve TSP-like problem (Pickups-first)
        # We must visit all pickups, then all deliveries.
        # This is two separate TSP problems.
        
        # Simple nearest-neighbor for pickups
        current_node = home_node
        while pickup_stops:
            best_stop_idx = -1
            min_dist = float('inf')
            
            for i, stop in enumerate(pickup_stops):
                # --- FIX: Use g.shortest_path_length ---
                dist = self.env.g.shortest_path_length(source=current_node, target=stop['node'], weight='length')
                if dist < min_dist:
                    min_dist = dist
                    best_stop_idx = i
            
            if best_stop_idx != -1:
                best_stop = pickup_stops.pop(best_stop_idx)
                route.add_stop(best_stop, self.env, (vehicle.capacity_weight, vehicle.capacity_volume))
                current_node = best_stop['node']
            else:
                break # Should not happen

        # Simple nearest-neighbor for deliveries
        while delivery_stops:
            best_stop_idx = -1
            min_dist = float('inf')
            
            for i, stop in enumerate(delivery_stops):
                # --- FIX: Use g.shortest_path_length ---
                dist = self.env.g.shortest_path_length(source=current_node, target=stop['node'], weight='length')
                if dist < min_dist:
                    min_dist = dist
                    best_stop_idx = i
            
            if best_stop_idx != -1:
                best_stop = delivery_stops.pop(best_stop_idx)
                route.add_stop(best_stop, self.env, (vehicle.capacity_weight, vehicle.capacity_volume))
                current_node = best_stop['node']
            else:
                break

        return route, new_inventory

    # --- ALNS Destroy Operators ---
    
    def random_route_removal(self, solution: Solution, num_to_remove: int) -> Tuple[List[str], List[str]]:
        """
        Removes `num_to_remove` routes from the solution.
        Returns the list of orders that were on those routes.
        """
        if not solution.routes:
            return [], []

        routes_to_remove = random.sample(
            list(solution.routes.keys()), 
            min(num_to_remove, len(solution.routes))
        )
        
        all_removed_orders = []
        for vehicle_id in routes_to_remove:
            route = solution.routes.pop(vehicle_id)
            all_removed_orders.extend(list(route.orders_served))
            solution.unassigned_orders.update(route.orders_served)
            
        solution.cost_calculated = False
        return all_removed_orders, routes_to_remove

    # --- ALNS Repair Operators ---

    def greedy_insertion_repair(self, solution: Solution, orders_to_insert: List[str], 
                                current_inventory: Dict, allow_new_routes: bool = False):
        """
        Inserts orders one by one into the best (cheapest) feasible position
        in an *existing* route.
        
        If allow_new_routes is True, it will create new routes if no
        insertion is found.
        """
        
        # We need to simulate inventory for this repair
        sim_inventory = copy.deepcopy(current_inventory)
        
        # First, re-build the inventory state based on the *current* solution
        for route in solution.routes.values():
            for stop in route.stops:
                if stop['type'] == 'pickup':
                    for sku, qty in stop['items'].items():
                        if sku in sim_inventory[stop['warehouse_id']]:
                            sim_inventory[stop['warehouse_id']][sku] -= qty
                        # We assume the initial solution was valid
        
        
        still_unassigned = []

        for order_id in orders_to_insert:
            if order_id not in solution.unassigned_orders:
                continue # Already assigned in a previous step
            
            order_reqs = self.order_reqs_cache[order_id]
            
            best_insertion = {
                'route': None,
                'cost_delta': float('inf'),
                'insertion_plan': None # (pickup_stops, delivery_stop, insert_idx)
            }
            
            # --- Try inserting into existing routes ---
            for route in solution.routes.values():
                insertion_plan = self._find_best_insertion_for_order(
                    route, order_id, order_reqs, sim_inventory
                )
                
                if insertion_plan:
                    cost_delta = insertion_plan['cost_delta']
                    if cost_delta < best_insertion['cost_delta']:
                        best_insertion = {
                            'route': route,
                            'cost_delta': cost_delta,
                            'insertion_plan': insertion_plan
                        }
            
            if best_insertion['route']:
                # --- Found a feasible insertion ---
                route = best_insertion['route']
                plan = best_insertion['insertion_plan']
                
                # Apply the insertion
                # We will just add the stops and *re-calculate* the
                # route's properties (weight, vol, dist) from scratch.
                
                # Let's use the plan from the *new* _find_best_insertion
                
                route.stops = plan['new_stops']
                route.current_weight = plan['new_weight']
                route.current_volume = plan['new_volume']
                route.current_distance = plan['new_distance']
                route.orders_served.add(order_id)
                
                # Update simulated inventory
                for wh_id, skus in plan['pickups_needed'].items():
                    for sku, qty in skus.items():
                        if sku in sim_inventory[wh_id]:
                            sim_inventory[wh_id][sku] -= qty
                        
                if order_id in solution.unassigned_orders:
                    solution.unassigned_orders.remove(order_id)
                solution.cost_calculated = False

            else:
                # No feasible insertion found in existing routes
                still_unassigned.append(order_id)
        
        # --- Handle orders that couldn't be inserted ---
        if still_unassigned and allow_new_routes:
            # Try to create new routes for them
            # This is basically the initial solution logic again
            
            # Find a free vehicle
            used_vehicles = set(solution.routes.keys())
            free_vehicles = [
                v for v in self.all_vehicles if v.id not in used_vehicles
            ]
            
            if free_vehicles:
                # Sort smallest first
                free_vehicles.sort(key=lambda v: v.capacity_weight)
                
                # Simple greedy pack for remaining
                orders_for_new_route = []
                current_w = 0.0
                current_v = 0.0
                vehicle = free_vehicles[0] # Try smallest free vehicle
                
                unassigned_after_pack = []
                
                for order_id in still_unassigned:
                    if order_id not in solution.unassigned_orders: continue

                    reqs = self.order_reqs_cache[order_id]
                    order_w = reqs['total_weight']
                    order_v = reqs['total_volume']
                    
                    if (current_w + order_w <= vehicle.capacity_weight and
                        current_v + order_v <= vehicle.capacity_volume):
                        
                        has_inv, _ = self._find_pickups_for_orders([order_id], sim_inventory)
                        if has_inv:
                            orders_for_new_route.append(order_id)
                            current_w += order_w
                            current_v += order_v
                        else:
                            unassigned_after_pack.append(order_id)
                    else:
                        unassigned_after_pack.append(order_id)
                
                if orders_for_new_route:
                    new_route, new_inv = self._build_route_for_orders(
                        vehicle, orders_for_new_route, sim_inventory
                    )
                    
                    if new_route:
                        new_route.finalize_route(self.env)
                        if new_route.current_distance <= vehicle.max_distance:
                            solution.add_route(new_route)
                            sim_inventory = new_inv # Commit inventory
                            # Remove these from still_unassigned
                            solution.unassigned_orders.difference_update(orders_for_new_route)
                        else:
                             if DEBUG:
                                print(f"Fallback route for {vehicle.id} failed max_dist")
            
    
    def _find_best_insertion_for_order(self, route: Route, order_id: str, 
                                     order_reqs: Dict, current_inventory: Dict) -> Optional[Dict]:
        """
        Finds the cheapest feasible insertion point for a single order
        into an existing route.
        
        Returns an 'insertion plan' dict if feasible, else None.
        """
        
        # --- FIX: Original AttributeError was here ---
        vehicle = self.env.get_vehicle_by_id(route.vehicle_id)
        # Corrected capacity check:
        max_weight = vehicle.capacity_weight
        max_volume = vehicle.capacity_volume
        max_distance = vehicle.max_distance

        order_w = order_reqs['total_weight']
        order_v = order_reqs['total_volume']

        # 1. Check if vehicle can *ever* hold this order
        # We need to check against *peak* load
        # This requires simulating the route...
        
        # 2. Check inventory
        has_inv, pickups_needed = self._find_pickups_for_orders([order_id], current_inventory)
        if not has_inv:
            return None # Not enough inventory in network
            
        # 3. Find best insertion point
        
        # Create new pickup stops
        new_pickup_stops = []
        for wh_id, items in pickups_needed.items():
            new_pickup_stops.append({
                'node': self.warehouse_nodes[wh_id],
                'type': 'pickup',
                'warehouse_id': wh_id,
                'items': items
            })
            
        # Create new delivery stop
        new_delivery_stop = {
            'node': self.order_locations[order_id],
            'type': 'delivery',
            'order_id': order_id,
            # --- FIX: Access the new 'skus' key from the cache ---
            'items': order_reqs['skus']
        }
        
        # Find first delivery stop in *existing* route
        first_delivery_idx = -1
        for i, stop in enumerate(route.stops):
            if stop['type'] == 'delivery':
                first_delivery_idx = i
                break
        
        if first_delivery_idx == -1:
             # Route has no deliveries yet (maybe just pickups?)
             first_delivery_idx = len(route.stops) -1 # Insert before home_end
             if first_delivery_idx <= 0: # Empty route
                 first_delivery_idx = 1
        
        
        best_plan = { 'cost_delta': float('inf'), 'new_stops': [], 'new_distance': 0.0 }
        
        # Try inserting the new delivery at all possible delivery positions
        # (i.e., after all pickups, before home_end)
        for i in range(first_delivery_idx, len(route.stops)):
            
            # --- Try inserting new pickups ---
            # Simplest: add all new pickups *before* all other stops
            # (This is suboptimal for routing but easier to validate)
            
            # We must not finalize the route before insertion
            if route.is_finalized():
                existing_stops = route.stops[:-1] # Remove home_end
            else:
                existing_stops = route.stops
            
            temp_stops = existing_stops[:1] + new_pickup_stops + existing_stops[1:]
            
            # Insert delivery stop
            # Adjust index for the added pickup stops
            temp_stops.insert(i + len(new_pickup_stops), new_delivery_stop)
            
            # --- Now, validate this new hypothetical route ---
            is_feasible, new_w, new_v, new_dist = self._validate_route_stops(
                temp_stops, vehicle, route.home_node
            )
            
            if is_feasible:
                cost_delta = new_dist - route.current_distance
                if cost_delta < best_plan['cost_delta']:
                    best_plan = {
                        'cost_delta': cost_delta,
                        # Add the home_end stop back
                        'new_stops': temp_stops + [{'node': route.home_node, 'type': 'home_end', 'items': {}}],
                        'new_weight': new_w,
                        'new_volume': new_v,
                        'new_distance': new_dist,
                        'pickups_needed': pickups_needed # To update inventory
                    }
        
        if best_plan['cost_delta'] < float('inf'):
            return best_plan
        else:
            return None # No feasible insertion found

    def _validate_route_stops(self, stops: List[Dict], vehicle: object, home_node: int) -> Tuple[bool, float, float, float]:
        """
        Checks a list of stops for feasibility (capacity, distance).
        Assumes `stops` list does NOT include the final home_end.
        Returns (is_feasible, peak_weight, peak_volume, total_distance)
        """
        # --- FIX: Original AttributeError was here ---
        max_weight = vehicle.capacity_weight
        max_volume = vehicle.capacity_volume
        max_distance = vehicle.max_distance
        
        current_w = 0.0
        current_v = 0.0
        peak_w = 0.0
        peak_v = 0.0
        total_dist = 0.0
        
        if not stops:
            return True, 0, 0, 0

        last_node = stops[0]['node'] # Should be home_start
        
        for stop in stops[1:]:
            new_node = stop['node']
            # --- FIX: Use g.shortest_path_length ---
            total_dist += self.env.g.shortest_path_length(source=last_node, target=new_node, weight='length')
            
            if stop['type'] == 'pickup':
                for sku, qty in stop['items'].items():
                    sku_details = self.env.get_sku_details(sku)
                    if sku_details:
                        current_w += sku_details['weight'] * qty
                        current_v += sku_details['volume'] * qty
                peak_w = max(peak_w, current_w)
                peak_v = max(peak_v, current_v)
            
            elif stop['type'] == 'delivery':
                # --- FIX: Access the new 'skus' key from the cache ---
                # We need the *cache* not the stop['items'] for this
                order_reqs = self.order_reqs_cache[stop['order_id']]
                for sku, qty in order_reqs['skus'].items():
                    sku_details = self.env.get_sku_details(sku)
                    if sku_details:
                        current_w -= sku_details['weight'] * qty
                        current_v -= sku_details['volume'] * qty
            
            if current_w < -0.01 or current_v < -0.01:
                # Should not happen if pickups are correct
                if DEBUG: print("Validation fail: Negative weight/volume")
                return False, 0, 0, 0
                
            if peak_w > max_weight or peak_v > max_volume:
                if DEBUG: print(f"Validation fail: Exceeded capacity. W: {peak_w:.2f}/{max_weight}, V: {peak_v:.2f}/{max_volume}")
                return False, 0, 0, 0 # Exceeded capacity
                
            last_node = new_node
        
        # Add final trip back to home
        # --- FIX: Use g.shortest_path_length ---
        total_dist += self.env.g.shortest_path_length(source=last_node, target=home_node, weight='length')
            
        if total_dist > max_distance:
            if DEBUG: print(f"Validation fail: Exceeded distance. {total_dist:.2f}/{max_distance}")
            return False, 0, 0, 0 # Exceeded distance

        return True, peak_w, peak_v, total_dist


# --- Solver Entry Point ---

def solver(env: LogisticsEnvironment) -> Dict[str, List[Dict]]:
    """
    Main solver function entry point for the hackathon.
    """
    
    # Make stochastic components reproducible
    random.seed(K_MEANS_SEED)
    if np is not None:
        np.random.seed(K_MEANS_SEED)

    # Initialize and run the ALNS solver
    alns_solver = ALNSSolver(env)
    
    try:
        best_solution = alns_solver.solve()
        
        # Convert final Solution object to dict
        solution_dict = best_solution.to_dict()
        
        # Final validation
        is_valid, msg = env.validate_solution_complete(solution_dict)
        if not is_valid:
            print(f"CRITICAL: Final ALNS solution is invalid: {msg}")
            # If invalid, try returning the *initial* solution
            initial_solution = alns_solver.generate_initial_solution()
            initial_solution.calculate_cost()
            if initial_solution.cost < float('inf'):
                print("Returning initial greedy solution instead.")
                return initial_solution.to_dict()
            else:
                print("Initial solution is also invalid. Returning empty.")
                return {} # Empty solution
        
        return solution_dict

    except Exception as e:
        print(f"An error occurred during the ALNS solve process: {e}")
        import traceback
        traceback.print_exc()
        
        print("Falling back to initial greedy solution...")
        try:
            # Try to return the initial solution as a fallback
            initial_solution = alns_solver.generate_initial_solution()
            initial_solution.calculate_cost()
            if initial_solution.cost < float('inf'):
                return initial_solution.to_dict()
            else:
                return {}
        except Exception as e2:
            print(f"Fallback solution failed: {e2}")
            return {} # Return empty solution if all else fails


if __name__ == "__main__":
    """
    For local testing and dashboard runs.
    """
    print("Running ALNS solver locally...")
    env = LogisticsEnvironment()
    
    # Set the solver
    env.set_solver(solver)

    # To run headlessly:
    print("Running headless...")
    results = env.run_headless("alns_solver_run")
    print("\n--- Solver Results ---")
    print(results)
    
    # To run with the dashboard (uncomment below):
    # print("\nLaunching dashboard...")
    # print("NOTE: Dashboard will re-run the solver.")
    # env.launch_dashboard()


