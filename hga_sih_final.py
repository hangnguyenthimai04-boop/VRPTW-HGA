# ==================================================
# HGA-SIH - CORRECTED FOR ALL INSTANCES
# ==================================================

from parsing import parse_file
from vrptw_functions import calculate_distances, check_time_windows, routes_distance, routes_time, begin_time
from plot_routes import plot_routes
import random
import time

# -------------------- SIH functions (unchanged) --------------------
def c11(i, u, j, d, mu):
    return d[i][u] + d[u][j] - mu * d[i][j]

def c12(u, j, route, t, customers):
    new_route = route[:]
    if j == 0:
        new_route.append(u)
    else:
        new_route.insert(route.index(j), u)
    return begin_time(j, new_route, t, customers) - begin_time(j, route, t, customers)

def insertion(route, u, pair):
    i, j = pair
    new_route = route[:]
    if i == 0:
        new_route.insert(0, u)
    elif j == 0:
        new_route.append(u)
    else:
        new_route.insert(new_route.index(j), u)
    return new_route

def insertion_heuristic(d, t, problem, customers, i1_params, init_criterium):
    mu, lam, alpha1, alpha2 = i1_params
    routes = []
    unrouted = list(range(1, len(customers)))
    while unrouted:
        if init_criterium == 0:
            seed = max(unrouted, key=lambda c: d[0][c])
        else:
            seed = min(unrouted, key=lambda c: customers[c].due_date)
        route = [seed]
        unrouted.remove(seed)
        load = customers[seed].demand
        while True:
            candidates = []
            for u in unrouted:
                if load + customers[u].demand > problem.capacity:
                    continue
                best_c1 = float('inf')
                best_pos = None
                positions = list(zip([0] + route, route + [0]))
                for i, j in positions:
                    new_route = insertion(route, u, (i, j))
                    if not check_time_windows(new_route, t, customers):
                        continue
                    c1_val = alpha1 * c11(i, u, j, d, mu) + alpha2 * c12(u, j, route, t, customers)
                    if c1_val < best_c1:
                        best_c1 = c1_val
                        best_pos = (i, j)
                if best_pos is not None:
                    c2_val = lam * d[0][u] - best_c1
                    candidates.append((u, best_pos, c2_val))
            if not candidates:
                break

                # 🔥 sort theo c2 giảm dần
            candidates.sort(key=lambda x: -x[2])

                # 🔥 chọn top-k
            top_k = candidates[:3]

                # 🔥 70% greedy, 30% random
            if random.random() < 0.7:
                u, (i, j), _ = top_k[0]
            else:
                u, (i, j), _ = random.choice(top_k)
            route = insertion(route, u, (i, j))
            unrouted.remove(u)
            load += customers[u].demand
        routes.append(route)
    return routes

# -------------------- Chromosome functions (FIXED) --------------------
def routes_to_chromosome(routes):
    return [c for r in routes for c in r]

def chromosome_to_routes(chromosome, problem, customers, t):
    routes = []
    route = []
    load = 0

    for u in chromosome:
        route.append(u)
        load += customers[u].demand

        if load > problem.capacity or not check_time_windows(route, t, customers):
            route.pop()  # rollback
            routes.append(route)

            route = [u]
            load = customers[u].demand

    if route:
        routes.append(route)

    return routes

def fitness(chrom, problem, customers, d, t):
    routes = chromosome_to_routes(chrom, problem, customers, t)
    num_routes = len(routes)
    dist = routes_distance(routes, d)

    penalty = 10000 * max(0, num_routes - problem.vehicle_number)

    return dist + penalty
# -------------------- Genetic operators --------------------
def pmx(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = p1[a:b]
    for i in range(a, b):
        if p2[i] not in child:
            pos = i
            while True:
                val = p1[pos]
                pos = p2.index(val)
                if child[pos] is None:
                    child[pos] = p2[i]
                    break
    for i in range(size):
        if child[i] is None:
            child[i] = p2[i]
    return child

def mutate(chrom):
    a, b = sorted(random.sample(range(len(chrom)), 2))
    chrom[a:b+1] = reversed(chrom[a:b+1])   # correct inversion
    return chrom

def tournament_selection(pop, fits, k=3):
    best_idx = None
    best_fit = float('inf')   # ✅

    for _ in range(k):
        idx = random.randrange(len(pop))
        if fits[idx] < best_fit:
            best_fit = fits[idx]
            best_idx = idx

    return pop[best_idx]

# -------------------- Initial population (diverse SIH) --------------------
def init_population(problem, customers, d, t, pop_size):
    param_sets = [(1,2,1,0), (1,1,1,0), (1,1,0,1), (1,2,0,1)]
    init_types = [0, 1]
    pop = []
    # 50% SIH with random params
    num_sih = pop_size // 2
    for _ in range(num_sih):
        params = random.choice(param_sets)
        itype = random.choice(init_types)
        routes = insertion_heuristic(d, t, problem, customers, params, itype)
        # Ensure routes are feasible (some SIH may produce infeasible)
        feasible = True
        for r in routes:
            if sum(customers[c].demand for c in r) > problem.capacity:
                feasible = False
                break
            if not check_time_windows(r, t, customers):
                feasible = False
                break
        if feasible:
            pop.append(routes_to_chromosome(routes))
        else:
            # fallback: random permutation
            chrom = list(range(1, len(customers)))
            random.shuffle(chrom)
            pop.append(chrom)
    # 50% random
    all_cust = list(range(1, len(customers)))
    while len(pop) < pop_size:
        chrom = all_cust[:]
        random.shuffle(chrom)
        pop.append(chrom)
    return pop

# -------------------- Local search (unchanged) --------------------
def route_distance(route, d):
    if not route:
        return 0
    dist = d[0][route[0]]
    for i in range(len(route)-1):
        dist += d[route[i]][route[i+1]]
    dist += d[route[-1]][0]
    return dist

def two_opt(route, d, customers, t):
    best = route[:]
    best_dist = route_distance(best, d)

    improved = True
    while improved:
        improved = False

        for i in range(len(route)-1):
            for j in range(i+1, len(route)):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]

                # 🔥 CHECK TIME WINDOW
                if not check_time_windows(new_route, t, customers):
                    continue

                new_dist = route_distance(new_route, d)

                if new_dist < best_dist:
                    best = new_route
                    best_dist = new_dist
                    improved = True

        route = best[:]

    return best

def relocate(routes, d, customers, t):
    best = [r[:] for r in routes]
    best_fit = (len(best), routes_distance(best, d))
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            for j in range(len(best)):
                if i == j:
                    continue
                for u in best[i]:
                    new_r1 = [x for x in best[i] if x != u]
                    for pos in range(len(best[j])+1):
                        new_r2 = best[j][:pos] + [u] + best[j][pos:]
                        if check_time_windows(new_r2, t, customers):
                            new_routes = [r[:] for r in best]
                            new_routes[i] = new_r1
                            new_routes[j] = new_r2
                            new_routes = [r for r in new_routes if r]
                            new_fit = (len(new_routes), routes_distance(new_routes, d))
                            if new_fit < best_fit:
                                best = new_routes
                                best_fit = new_fit
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best

def swap(routes, d, customers, t):
    best = [r[:] for r in routes]
    best_fit = (len(best), routes_distance(best, d))
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            for j in range(i+1, len(best)):
                for u in best[i]:
                    for v in best[j]:
                        new_r1 = best[i][:]
                        new_r2 = best[j][:]
                        idx_u = new_r1.index(u)
                        idx_v = new_r2.index(v)
                        new_r1[idx_u] = v
                        new_r2[idx_v] = u
                        if check_time_windows(new_r1, t, customers) and check_time_windows(new_r2, t, customers):
                            new_routes = [r[:] for r in best]
                            new_routes[i] = new_r1
                            new_routes[j] = new_r2
                            new_fit = (len(new_routes), routes_distance(new_routes, d))
                            if new_fit < best_fit:
                                best = new_routes
                                best_fit = new_fit
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best

# -------------------- Thêm hàm merge_routes --------------------
def try_merge(routes, customers, problem, t):

    for i in range(len(routes)):
        for j in range(i+1, len(routes)):

            merged = routes[i] + routes[j]

            load = sum(customers[u].demand for u in merged)
            if load > problem.capacity:
                continue

            if not check_time_windows(merged, t, customers):
                continue

            new_routes = routes[:]
            new_routes[i] = merged
            new_routes.pop(j)

            return new_routes

    return routes
# -------------------- Cập nhật local_search --------------------
def local_search(routes, d, customers, t, problem):

    routes = [two_opt(r, d, customers, t) for r in routes]

    for _ in range(5):  # 🔥 loop để ép giảm xe
        new_routes = try_merge(routes, customers, problem, t)
        if len(new_routes) == len(routes):
            break
        routes = new_routes

    routes = relocate(routes, d, customers, t)
    routes = swap(routes, d, customers, t)

    return routes


# -------------------- Tăng quy mô cho HGA_SIH --------------------
def HGA_SIH(problem, customers, d, t, pop_size=1000, generations=200):
    pop = init_population(problem, customers, d, t, pop_size)
    best_chrom = None
    best_fit = float('inf')   # ✅

    for gen in range(generations):
        fits = [fitness(ch, problem, customers, d, t) for ch in pop]
        # Cập nhật best
        for i, fit in enumerate(fits):
            if fit < best_fit:
                best_fit = fit
                best_chrom = pop[i][:]
        # Elitism: giữ 2 cá thể tốt nhất (thay vì 1) để tăng hội tụ
        elite_size = max(2, int(pop_size * 0.05))
        sorted_indices = sorted(range(len(pop)), key=lambda i: fits[i])
        elites = [pop[i] for i in sorted_indices[:elite_size]]
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits, k=3)   # giảm k xuống 3
            p2 = tournament_selection(pop, fits, k=3)
            child = pmx(p1, p2)
            if random.random() < 0.1:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop

    best_routes = chromosome_to_routes(best_chrom, problem, customers, t)
    return best_routes

# -------------------- Cập nhật run_instance để gọi local_search với problem --------------------
def run_instance(instance_path, runs=30):
    problem, customers = parse_file(instance_path)

    print(f"\n--- {problem.problem_id} ---")
    print(f"Capacity={problem.capacity}, Vehicles allowed={problem.vehicle_number}")

    d = calculate_distances(customers)

    best_routes = None
    best_fit = float('inf')

    for run in range(runs):

        # seed khác nhau mỗi lần chạy
        random.seed(run * 1234567)

        routes = HGA_SIH(problem, customers, d, d, pop_size=100, generations=50)

        # local search mạnh hơn
        for _ in range(3):
            routes = local_search(routes, d, customers, d, problem)
            
        # 🔥 VALIDATION FULL
        for r in routes:
            if not check_time_windows(r, d, customers):
                print("❌ TIME WINDOW FAIL")

            load = sum(customers[c].demand for c in r)
            if load > problem.capacity:
                print("❌ CAPACITY FAIL")

        # 👉 tách riêng vehicles và distance
        vehicles = len(routes)
        distance = routes_distance(routes, d)

        # 👉 fitness có penalty
        fit = distance + 10000 * max(0, vehicles - problem.vehicle_number)

        # update best
        if fit < best_fit:
            best_fit = fit
            best_routes = routes

        print(f"Run {run+1:2d}: vehicles={vehicles:2d}, distance={distance:8.2f}")

    # 👉 tính lại best để in
    best_vehicles = len(best_routes)
    best_distance = routes_distance(best_routes, d)

    print(f"\n*** Best: vehicles={best_vehicles}, distance={best_distance:.2f} ***")

    title = f"{problem.problem_id} | vehicles={best_vehicles} | dist={round(best_distance,2)}"

    plot_routes(customers, best_routes, customers[0], d, title=title)

    return best_routes
if __name__ == "__main__":
    # Test C101 (should get 828.94)
    run_instance("data/r105.txt", runs=3)
    # Uncomment for other instances
    #run_instance("data/c201.txt", runs=30)