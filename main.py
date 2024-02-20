from copy import copy
import queue
import time

BOARD_LENGTH = 5
PLAYERS_PLAYING = 2
HOR = 0
VER = 1
DEPTH = 4


class Node:
    def __init__(self, parent, action, state=None):
        self.parent = parent
        self.state = state
        self.action = action
        self.depth = 0
        if state is not None:
            self.state = state
        if parent is not None and action is not None:
            self.state = parent.state.make_action(action)
            self.parent = parent
            self.depth = parent.depth + 1

        self.dists_to_goal = {pid: self.state.dist_to_goal_astar(self.state.players[pid], return_inf=True)
                              for pid in range(len(self.state.players))}
        self.children = []
        self.info = None

    def expand(self):
        moves = list(self.state.get_legal_moves_from(*self.state.players[self.state.player_to_move_index].pos))
        blocks = list(self.state.get_legal_blocks())
        for action in moves + blocks:
            self.children.append(Node(self, action))

    def alpha_value_for_pid(self, pid, stop_after=None):
        if self.depth >= DEPTH:
            if self.state.players[pid].has_won():
                return float('-infinity')
            if 0 in set(self.dists_to_goal.values()):
                return float('infinity')
            return self.dists_to_goal[pid] - (1 if self.state.player_to_move_index == pid else 0) - \
                min(self.dists_to_goal[enemy] for enemy in range(len(self.state.players)) if enemy != pid)
        else:
            if len(self.children) == 0:
                self.expand()
            if self.state.player_to_move_index == pid:
                return min([c.alpha_value_for_pid(pid) for c in self.children])
            else:
                # return min([c.alpha_value_for_pid(pid) for c in self.children])
                beta = float('-infinity')
                for c in self.children:
                    b = c.alpha_value_for_pid(pid, stop_after=stop_after)
                    if stop_after is not None and b > stop_after:
                        return b
                    if b > beta:
                        beta = b
                return beta


class Player:
    def __init__(self, id):
        self.id = id
        self.num_of_boards = 20 / PLAYERS_PLAYING
        if id == 0:
            self.pos = (int(BOARD_LENGTH / 2), 0)
        elif id == 1:
            self.pos = (0, int(BOARD_LENGTH / 2))
        elif id == 2:
            self.pos = (int(BOARD_LENGTH / 2), BOARD_LENGTH - 1)
        elif id == 3:
            self.pos = (BOARD_LENGTH - 1, int(BOARD_LENGTH / 2))

    def has_won(self):
        if self.id == 0:
            return self.pos[1] == BOARD_LENGTH - 1
        elif self.id == 1:
            return self.pos[0] == BOARD_LENGTH - 1
        elif self.id == 2:
            return self.pos[1] == 0
        elif self.id == 3:
            return self.pos[0] == 0

    def __copy__(self):
        clone = Player(self.id)
        clone.num_of_boards = self.num_of_boards
        clone.pos = self.pos
        return clone

    def dist_to_edge(self):
        if self.id == 0:
            return BOARD_LENGTH - 1 - self.pos[1]
        elif self.id == 1:
            return BOARD_LENGTH - 1 - self.pos[0]
        elif self.id == 2:
            return self.pos[1]
        elif self.id == 3:
            return self.pos[0]


class State:
    def __init__(self, num_of_players=PLAYERS_PLAYING):
        if num_of_players == 2:
            self.players = [Player(0), Player(2)]
        else:
            self.players = [Player(0), Player(1), Player(2), Player(3)]
        self.player_to_move_index = 0
        self.blocked = set()
        self.dists = {pid: None for pid in range(num_of_players)}

    def make_action(self, action, want_print=False):
        if isinstance(action, tuple):
            new_state = copy(self)
            new_state.get_player().pos = (action[0], action[1])
            new_state.next_player()
        else:
            blocks = set(action)
            new_state = copy(self)
            new_state.get_player().num_of_boards -= 1
            new_state.next_player()
            new_state.blocked = new_state.blocked.union(blocks)
        if want_print:
            print(f'Action {action} taken.')
        return new_state

    def next_player(self):
        self.player_to_move_index = (self.player_to_move_index + 1) % len(self.players)

    def __copy__(self):
        clone = State(len(self.players))
        clone.player_to_move_index = self.player_to_move_index
        clone.blocked = self.blocked.copy()
        for p_id in range(len(self.players)):
            clone.players[p_id].pos = self.players[p_id].pos
            clone.players[p_id].num_of_boards = self.players[p_id].num_of_boards

        return clone

    def get_player(self):
        return self.players[self.player_to_move_index]

    def tile_is_taken(self, row, col):
        return not all(a.pos != (row, col) for a in self.players)

    def dist_to_goal_astar(self, player, return_inf=False):
        pid = self.players.index(player)
        if self.dists[pid] is not None:
            return self.dists[pid]
        dists = {player.pos: 0}
        q = queue.PriorityQueue()
        q.put((0, player.pos))
        while q.qsize() > 0:
            pos = q.get()[1]
            row = pos[0]
            col = pos[1]
            for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                row_ = row + d[0]
                col_ = col + d[1]
                if row_ < 0 or row_ >= BOARD_LENGTH or col_ < 0 or col_ >= BOARD_LENGTH:
                    continue
                if ((row, col), (row_, col_)) in self.blocked or ((row_, col_), (row, col)) in self.blocked:
                    continue
                if not (row_, col_) in dists:
                    dists[(row_, col_)] = dists[(row, col)] + 1
                    p_clone = copy(player)
                    p_clone.pos = (row_, col_)
                    if p_clone.has_won():
                        return dists[(row_, col_)]
                    q.put((p_clone.dist_to_edge() + dists[(row_, col_)], (row_, col_)))
        return None if not return_inf else float('inf')

    def get_legal_moves_from(self, row, col):
        legal_moves = set()
        for d in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            if not (((row, col), (row + d[0], col + d[1])) in self.blocked
                    or ((row + d[0], col + d[1]), (row, col)) in self.blocked):
                if not self.tile_is_taken(row + d[0], col + d[1]):
                    legal_moves.add((row + d[0], col + d[1]))
                else:
                    if not (((row + d[0], col + d[1]), (row + 2 * d[0], col + 2 * d[1])) in self.blocked
                            or ((row + 2 * d[0], col + 2 * d[1]), (row + d[0], col + d[1])) in self.blocked):
                        legal_moves.add((row + 2 * d[0], col + 2 * d[1]))
                    else:
                        if d in ((0, 1), (0, -1)):
                            new_ds = ((1, 0), (-1, 0))
                        else:
                            new_ds = ((0, 1), (0, -1))
                        for new_d in new_ds:
                            if not (((row + d[0], col + d[1]),
                                     (row + d[0] + new_d[0], col + d[1] + new_d[1])) in self.blocked
                                    or ((row + d[0] + new_d[0], col + d[1] + new_d[1]),
                                        (row + d[0], col + d[1])) in self.blocked):
                                legal_moves.add((row + d[0] + new_d[0], col + d[1] + new_d[1]))
        filtered_lms = set()
        for move in legal_moves:
            if (0 <= move[0] < BOARD_LENGTH and 0 <= move[1] < BOARD_LENGTH and
                    not self.tile_is_taken(move[0], move[1])):
                filtered_lms.add(move)
        return filtered_lms

    def block_doesnt_cut_anyone_off(self, block):
        clone = copy(self)
        clone.blocked = clone.blocked.union(block)
        for p in clone.players:
            if clone.dist_to_goal_astar(p) is None:
                return False
        return True

    def get_legal_blocks(self):
        legal_blocks = []
        for row in range(BOARD_LENGTH - 1):
            for col in range(BOARD_LENGTH - 1):
                blocks = ({((row, col), (row + 1, col)), ((row, col + 1), (row + 1, col + 1))},
                          {((row, col), (row, col + 1)), ((row + 1, col), (row + 1, col + 1))})
                for block in blocks:
                    if len(self.blocked.intersection(block)) == 0 and self.block_doesnt_cut_anyone_off(block):
                        legal_blocks.append(block)
        return legal_blocks

    def get_move_from_input(self):
        new_state = None
        lms = self.get_legal_moves_from(*self.get_player().pos)
        no_moves = len(lms) == 0

        if no_moves:
            print('No possible move. Skipping.')
            row, col = self.get_player().pos
        else:
            print(f'Where do you want to move?')
            try:
                row = int(input('row:'))
                col = int(input('col:'))
            except:
                return None
        if no_moves or (row, col) in lms:
            new_state = copy(self)
            new_state.get_player().pos = (row, col)
            new_state.next_player()
        return new_state

    def get_blocks_from_input(self):
        new_state = None
        if input('Do you want to put a board vertical?: ') in ('1', 'yes', 'y', 'Y', 'Yes'):
            orient = VER
        else:
            orient = HOR
        try:
            print(f'Between which' + (' rows?' if orient == HOR else ' cols?'))
            if orient == HOR:
                row1, row2 = sorted((int(input('row1:')), int(input('row2:'))))
                print('At which cols?')
            col1, col2 = sorted((int(input('col1:')), int(input('col2:'))))
            if orient == VER:
                print('At which rows?')
                row1, row2 = sorted((int(input('row1:')), int(input('row2:'))))
        except:
            print('\nIllegal input!\n')
            return None

        if orient == VER:
            blocks = {((row1, col1), (row1, col2)), ((row2, col1), (row2, col2))}
        else:
            blocks = {((row1, col1), (row2, col1)), ((row1, col2), (row2, col2))}
        if blocks in self.get_legal_blocks():
            new_state = copy(self)
            new_state.get_player().num_of_boards -= 1
            print(f'Player {new_state.player_to_move_index}, you have {new_state.get_player().num_of_boards} left.')
            new_state.next_player()
            new_state.blocked = new_state.blocked.union(blocks)

        return new_state

    def get_action_from_input(self):
        print(f'Player {self.player_to_move_index} make action.')
        no_boards = self.get_player().num_of_boards <= 0
        if no_boards:
            print('Not enough boards to not move')
        if (no_boards or
                input('Do you want to move?: ') in ('1', 'yes', 'y', 'Y', 'Yes')):
            return self.get_move_from_input()
        else:
            return self.get_blocks_from_input()

    def __str__(self):
        board = '  '
        for col in range(BOARD_LENGTH):
            board += ' ' + str(col) + '  '
        board += '\n'
        for row in range(BOARD_LENGTH):
            board += str(row) + '|'
            for col in range(BOARD_LENGTH):
                for player in self.players:
                    if player.pos == (row, col):
                        board += ' ' + str(self.players.index(player)) + ' '
                if not self.tile_is_taken(row, col):
                    board += '   '
                if ((row, col), (row, col + 1)) in self.blocked:
                    board += '|'
                else:
                    board += ' '
            board += '|\n |'

            for col in range(BOARD_LENGTH):
                if ((row, col), (row + 1, col)) in self.blocked:
                    board += ' - '
                else:
                    board += '   '
                board += '+'
            board += '|\n'
        return board


def alpha_beta_search(state):
    pid = state.player_to_move_index
    root = Node(None, None, state)
    node = root
    node.expand()
    best_a = float('infinity')
    best_child = node.children[0]
    for c in node.children:
        a = c.alpha_value_for_pid(pid, stop_after=best_a)
        if a < best_a:
            print(f'A: {a}')
            best_a = a
            best_child = c
    return best_child.action


def play_game_from_input():
    state = State(2)
    while all([not p.has_won() for p in state.players]):
        print(str(state))
        new_state = state.get_action_from_input()
        if new_state is not None:
            state = new_state
    print(f'Player {state.player_to_move_index} won!')


if __name__ == '__main__':
    state = State(PLAYERS_PLAYING)
    while all(not p.has_won() for p in state.players):
        print(str(state))
        print(f'Turn: {state.player_to_move_index}')
        print(f"boards left:  {[p.num_of_boards for p in state.players]}")
        time.sleep(2)
        state = state.make_action(alpha_beta_search(state), want_print=True)
    print(str(state))
    for pid in range(len(state.players)):
        if state.players[pid].has_won():
            print(f'Player {pid} has won!')
