import heapq

def get_distance(a, b):
    # Odległość euklidesowa (pierwiastek z sumy kwadratów)
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def astar(grid, start, goal, limit=20000): # Zwiększamy nieco limit dla dużych map kosztów
    rows = len(grid)
    cols = len(grid[0])

    open_set = []
    # heapq potrzebuje krotki: (f_score, wezel)
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        if current in closed_set:
            continue
        closed_set.add(current)

        if len(closed_set) > limit: 
            break

        # Przeszukiwanie 8 kierunków
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # USUNIĘTO: if grid[neighbor[0]][neighbor[1]] == 1: continue
                # Ponieważ teraz 1 to wolna przestrzeń, a nie ściana!

                # Koszt ruchu = odległość geometryczna * waga terenu (1, 5 lub 10)
                move_cost = get_distance(current, neighbor) * grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + get_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return None # Zwróci None tylko, jeśli autentycznie brakuje pamięci/limitu lub cel jest poza mapą