from searching_framework import Problem, astar_search


class Climbing(Problem):
    def __init__(self, initial, allowed):
        super().__init__(initial)
        self.allowed = allowed
        self.width = 5;
        self.height = 9

    def successor(self, state):
        successors = {}
        man_x, man_y = state[0]
        direction = state[1]
        house_x, house_y = state[2]

        if direction == 'right':
            if house_x + 1 < self.width:
                new_house_x = house_x + 1
                new_dir = 'right'
            else:
                new_house_x = house_x - 1
                new_dir = 'left'
        else:
            if house_x - 1 >= 0:
                new_house_x = house_x - 1
                new_dir = 'left'
            else:
                new_house_x = house_x + 1
                new_dir = 'right'

        new_house_pos = (new_house_x, house_y)

        moves = [
            ('Stay', (0, 0)),
            ('Up 1', (0, 1)),
            ('Up 2', (0, 2)),
            ('Up-right 1', (1, 1)),
            ('Up-right 2', (2, 2)),
            ('Up-left 1', (-1, 1)),
            ('Up-left 2', (-2, 2))
        ]

        for action, (dx, dy) in moves:
            new_x = man_x + dx
            new_y = man_y + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                if new_y == self.height - 1:
                    if (new_x, new_y) == new_house_pos:
                        successors[action] = ((new_x, new_y), new_dir, new_house_pos)
                else:
                    if (new_x, new_y) in self.allowed or action == 'Stay':
                        successors[action] = ((new_x, new_y), new_dir, new_house_pos)

        return successors

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        return self.successor(state)[action]

    def goal_test(self, state):
        return state[0] == state[2]

    def h(self, node):
        man_x, man_y = node.state[0]
        house_x, house_y = node.state[2]
        vertical_dist = abs(man_y - house_y) // 2
        horizontal_dist = abs(man_x - house_x) // 2
        return max(vertical_dist, horizontal_dist)


if __name__ == '__main__':
    allowed = [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (0, 2), (2, 2), (4, 2), (1, 3), (3, 3), (4, 3),
               (0, 4), (2, 4), (2, 5), (3, 5), (0, 6), (2, 6), (1, 7), (3, 7)]

    man_coord = input().split(",")
    man_pos = (int(man_coord[0]), int(man_coord[1]))
    house_coord = input().split(",")
    house_pos = (int(house_coord[0]), int(house_coord[1]))
    direction = input()

    climb = Climbing((man_pos, direction, house_pos), allowed)
    solution = astar_search(climb)

    if solution is not None:
        print(solution.solution())
    else:
        print('No solution!')
