from searching_framework import Problem, astar_search


class Labyrinth(Problem):
    def __init__(self, initial, walls, size, goal):
        super().__init__(initial, goal)
        self.walls = walls
        self.size = size

    def successor(self, state):
        successors = {}
        man_x, man_y = state
        for i in range(1, 4):
            if (man_x + i, man_y) not in self.walls and man_x + i < self.size:
                if i != 1:
                    successors[f'Right {i}'] = (man_x + i, man_y)
            else:
                break
        if man_y + 1 < self.size and (man_x, man_y + 1) not in self.walls:
            successors['Up'] = (man_x, man_y + 1)
        if man_y - 1 >= 0 and (man_x, man_y - 1) not in self.walls:
            successors['Down'] = (man_x, man_y - 1)
        if man_x - 1 >= 0 and (man_x - 1, man_y) not in self.walls:
            successors['Left'] = (man_x - 1, man_y)
        return successors

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        return self.successor(state)[action]

    def goal_test(self, state):
        return state == self.goal

    def h(self, node):
        x, y = node.state
        goal_x, goal_y = self.goal

        x_diff = goal_x - x
        y_diff = goal_y - y

        if x_diff > 0:
            h_cost = (x_diff // 3)
            remainder = x_diff % 3

            if remainder == 1:

                h_cost += 1
            elif remainder == 2:
                h_cost += 1
        else:
            h_cost = abs(x_diff)

        v_cost = abs(y_diff)

        return h_cost + v_cost


if __name__ == '__main__':
    size = int(input())
    num_walls = int(input())
    walls = list()

    for _ in range(num_walls):
        coords = input().split(",")
        walls.append((int(coords[0]), int(coords[1])))

    man_coords = input().split(",")
    man_position = (int(man_coords[0]), int(man_coords[1]))
    house_coords = input().split(",")
    house_position = (int(house_coords[0]), int(house_coords[1]))

    labyrinth = Labyrinth(man_position, walls, size, house_position)

    solution = astar_search(labyrinth)
    if solution is not None:
        print(solution.solution())
    else:
        print('No solution')
