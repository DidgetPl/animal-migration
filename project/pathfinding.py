import heapq


def get_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def astar(grid, start, goal, limit=20000):
    rows = len(grid)
    cols = len(grid[0])

    open_set = []
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

        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                move_cost = get_distance(current, neighbor) * grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + get_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None