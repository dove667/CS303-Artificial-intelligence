import numpy as np
import time
from typing import List, Tuple, Optional
from numba import njit


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
INFINITY = float('inf')

# 首次 JIT 预热标记（避免重复预热）
_NUMBA_WARMED_UP = False

DIRS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1) ,          (0, 1),
    (1, -1) , (1, 0),  (1, 1)
], dtype=np.int8)

# 不同阶段的启发式评估权重
# w1: 棋盘权重  w2: 稳定子  w3: 棋子数量  w4: 行动力
HURISTIC_WEIGHTS = {
    'begin': (3, 5, 1, 1),
    'middle': (3, 2, 2, 3),
    'end': (3, 1, 5, 1)
}

@njit(cache=True)
def nb_get_possible_moves(board, color):
    """
    返回所有“与对手棋子相邻且为空”的点，形状 (K,2) 的数组。
    """
    opp = (board == -color)
    neighbor = np.zeros_like(opp)
    # 上下左右
    neighbor[:-1, :] |= opp[1:, :] #上
    neighbor[1:,  :] |= opp[:-1, :] #下
    neighbor[:, :-1] |= opp[:, 1:] #左
    neighbor[:, 1: ] |= opp[:, :-1] #右
    # 对角
    neighbor[:-1, :-1] |= opp[1:, 1:]
    neighbor[:-1, 1: ] |= opp[1:, :-1]
    neighbor[1:,  :-1] |= opp[:-1, 1:]
    neighbor[1:,  1: ] |= opp[:-1, :-1]

    possible = neighbor & (board == 0)
    rows, cols = np.nonzero(possible) # 返回非零元素的坐标
    k = rows.shape[0]
    coords = np.empty((k, 2), dtype=np.int8)
    coords[:, 0] = rows
    coords[:, 1] = cols
    return coords

@njit(cache=True)
def nb_is_valid_move(board, row, col, color):
    """
    判断合法性并返回需要翻转的坐标数组 flips，形状 (M,2)。
    无翻子则返回 (False, 空数组)。
    """
    n = board.shape[0]
    if board[row, col] != 0:
        return False, np.empty((0, 2), dtype=np.int8)

    flips = np.empty((64, 2), dtype=np.int8)  
    idx = 0
    for dr, dc in DIRS:
        r = row + dr; c = col + dc
        # 第一格必须是对手棋子
        if r < 0 or r >= n or c < 0 or c >= n or board[r, c] != -color:
            continue
        count = 0
        while 0 <= r < n and 0 <= c < n and board[r, c] == -color:
            count += 1
            r += dr; c += dc
        if count > 0 and 0 <= r < n and 0 <= c < n and board[r, c] == color:
            # 回填需要翻转的格子
            for t in range(1, count + 1):
                rr = row + dr * t
                cc = col + dc * t
                flips[idx, 0] = rr
                flips[idx, 1] = cc
                idx += 1
    if idx == 0:
        return False, np.empty((0, 2), dtype=np.int8)
    return True, flips[:idx]

@njit(cache=True)
def nb_has_any_valid_move(chessboard, color) -> bool:
    """只要发现一个合法步就返回 True，减少不必要计算"""
    for r, c in nb_get_possible_moves(chessboard, color):
        ok, flips = nb_is_valid_move(chessboard, r, c, color)
        if ok and flips.shape[0] > 0:
            return True
    return False

@njit(cache=True)
def nb_count_legal_moves(chessboard, color) -> int:
    """返回合法步数量（行动子）"""
    cnt = 0
    for r, c in nb_get_possible_moves(chessboard, color):
        ok, flips = nb_is_valid_move(chessboard, r, c, color)
        if ok and flips.shape[0] > 0:
            cnt += 1
    return cnt

@njit(cache=True)
def nb_flip_inplace(board, flips, color):
    """把 flips 列表对应格子设置为 color（用于做/撤销）"""
    for i in range(flips.shape[0]):
        r = flips[i, 0]
        c = flips[i, 1]
        board[r, c] = color

@njit(cache=True)
def nb_candidates_with_flips_csr(board, color):
    """
    获取所有合法落子点及对应翻转棋子位置，采用 CSR-like 存储格式：
    - 把所有子列表元素按顺序拼在一个一维/二维缓冲区 flips_buf 里。
    - 用 offsets 记录每个子列表的起止位置：第 i 个子列表为 flips_buf[offsets[i] : offsets[i+1]]。

    返回:
        moves: (K,2) 合法落子
        flips_buf: (T,2) 所有翻子拼接
        offsets: (K+1,) CSR-like 索引
    这样每层只需一次 Python→Numba 边界调用。
    """
    possible = nb_get_possible_moves(board, color)
    K = possible.shape[0]
    # 先统计合法走法与总翻子数
    cand_tmp = np.empty((K, 2), dtype=np.int8)
    valid = 0 # 合法走法计数
    total = 0 # 总翻子计数
    for i in range(K):
        r = possible[i, 0]; c = possible[i, 1]
        ok, flips = nb_is_valid_move(board, r, c, color)
        if ok and flips.shape[0] > 0:
            cand_tmp[valid, 0] = r
            cand_tmp[valid, 1] = c
            total += flips.shape[0]
            valid += 1
    moves = cand_tmp[:valid].copy()
    flips_buf = np.empty((total, 2), dtype=np.int8)
    offsets = np.empty(valid + 1, dtype=np.int32)
    offsets[0] = 0
    write_ptr = 0
    vi = 0
    for i in range(K):
        r = possible[i, 0]; c = possible[i, 1]
        ok, flips = nb_is_valid_move(board, r, c, color)
        if ok and flips.shape[0] > 0:
            k = flips.shape[0]
            flips_buf[write_ptr:write_ptr + k, :] = flips
            write_ptr += k
            vi += 1
            offsets[vi] = write_ptr
    return moves, flips_buf, offsets

#don’t change the class name
class AI(object):
    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color , time_out=4.9):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm’s run
        # time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list.
        # The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.directions = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),          (0, 1),
                           (1, -1),  (1, 0), (1, 1)]
        self.max_depth = 64
        self.weighted_board = np.array([
            [1, 8, 3, 7, 7, 3, 8, 1],
            [8, 3, 2, 5, 5, 2, 3, 8],
            [3, 2, 6, 6, 6, 6, 2, 3],
            [7, 5, 6, 4, 4, 6, 5, 7],
            [7, 5, 6, 4, 4, 6, 5, 7],
            [3, 2, 6, 6, 6, 6, 2, 3],
            [8, 3, 2, 5, 5, 2, 3, 8],
            [1, 8, 3, 7, 7, 3, 8, 1],
        ], dtype=np.int8)
        # 用于各层走法排序的主变线提示：pv_moves[ply] = 该层推荐走法(根为0层)
        self.pv_moves: List[Optional[Tuple[int,int]]] = [None] * (self.max_depth)

        # 预热 Numba，避免首次搜索时编译耗时
        global _NUMBA_WARMED_UP
        if not _NUMBA_WARMED_UP:
            try:
                self._warmup_numba()
            except Exception:
                pass
            _NUMBA_WARMED_UP = True
            
    def _warmup_numba(self):
        """触发 numba 编译（在构造时，不占用走子时间）"""
        b = np.zeros((8, 8), dtype=np.int8)
        b[3, 3] = COLOR_WHITE; b[4, 4] = COLOR_WHITE
        b[3, 4] = COLOR_BLACK; b[4, 3] = COLOR_BLACK

        # 两方各跑一遍，触发所有内核编译
        for color in (COLOR_BLACK, COLOR_WHITE):
            _ = nb_get_possible_moves(b, color)
            _ = nb_count_legal_moves(b, color)
            moves, flips_buf, offsets = nb_candidates_with_flips_csr(b, color)
            if moves.shape[0] > 0:
                r0 = int(moves[0, 0]); c0 = int(moves[0, 1])
                ok, flips = nb_is_valid_move(b, r0, c0, color)
                if ok and flips.shape[0] > 0:
                    # 模拟一次翻转与回退
                    b[r0, c0] = color
                    nb_flip_inplace(b, flips, color)
                    nb_flip_inplace(b, flips, -color)
                    b[r0, c0] = COLOR_NONE

    def get_candidate_reversed_list(self, chessboard, color) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """
        获取所有合法落子点列表和对应的翻转棋子位置列表
        """
        moves, flips_buf, offsets = nb_candidates_with_flips_csr(chessboard, color)
        k = moves.shape[0]
        candidate_list: List[Tuple[int,int]] = [(int(moves[i, 0]), int(moves[i, 1])) for i in range(k)]
        # 懒切片视图，不复制
        reversed_list: List[np.ndarray] = [flips_buf[offsets[i]:offsets[i+1], :] for i in range(k)]
        return candidate_list, reversed_list
    
    # def _get_stable_disk(self, chessboard) -> int:
    #     """
    #     计算颜色的稳定子得分
    #     但是目前只是一个不准确的估计。
    #     """
    #     stable_coords = set()
    #     corners = [(0, 0), (0, self.chessboard_size - 1),
    #             (self.chessboard_size - 1, 0), (self.chessboard_size - 1, self.chessboard_size - 1)]
    #     corner_dirs = {
    #         (0, 0): [(0, 1), (1, 0), (1, 1)],
    #         (0, self.chessboard_size - 1): [(0, -1), (1, 0), (1, -1)],
    #         (self.chessboard_size - 1, 0): [(0, 1), (-1, 0), (-1, 1)],
    #         (self.chessboard_size - 1, self.chessboard_size - 1): [(0, -1), (-1, 0), (-1, -1)],
    #     }
    #     for cr, cc in corners:
    #         if chessboard[cr, cc] == COLOR_NONE:
    #             continue
    #         color = chessboard[cr, cc]
    #         for dr, dc in corner_dirs[(cr, cc)]:
    #             r, c = cr, cc
    #             while 0 <= r < self.chessboard_size and 0 <= c < self.chessboard_size and chessboard[r, c] == color:
    #                 stable_coords.add((r, c))
    #                 r += dr
    #                 c += dc
    #     return sum(int(chessboard[r, c]) for (r, c) in stable_coords)
        
    def evaluate(self, chessboard) -> float:
        """
        反黑白棋启发式评估（“白方优势”为正，“黑方优势”为负）。
        """
        # 棋盘权重
        weighted_chessboard = float(np.sum(self.weighted_board * chessboard))

        # 行动力
        white_cnt = nb_count_legal_moves(chessboard, COLOR_WHITE)
        black_cnt = nb_count_legal_moves(chessboard, COLOR_BLACK)
        mobility = float(white_cnt - black_cnt)

        # 棋子数量
        piece_count = int(np.sum(chessboard != COLOR_NONE))

        # 动态权重
        if piece_count <= 15:
            w1, w2, w3, w4 = HURISTIC_WEIGHTS['begin']
        elif piece_count <= 40:
            w1, w2, w3, w4 = HURISTIC_WEIGHTS['middle']
        else:
            w1, w2, w3, w4 = HURISTIC_WEIGHTS['end']

        score = w1 * weighted_chessboard + w2 * 0.0 + w3 * piece_count + w4 * mobility
        return score

    def minimax_search(self, chessboard, depth_limit, deadline) -> Tuple[float, Tuple[int, int], bool, List[Tuple[int,int]]]:
        """
        黑棋（-1）为max节点，白棋（+1）为min节点。
        评估函数以“白方优势”为正数，故该极大极小方向在反黑白棋下自洽。
        Args:
            chessboard: 当前棋盘状态
            candidate_list: 当前可选的落子点列表
        Returns:
            best_score: 最佳走法对应的评估值
            best_move: 最佳走法
            timed_out: 是否因超时而中断
            pv: 主变线走法列表
        备注：
            deadline: 本次搜索的截止时间点（time.time()）
        """
        def time_exceeded() -> bool:
            return time.time() > deadline

        def reorder_with_previous(cands, revs, depth):
            # 在每一层用上一轮的主变线置顶
            move = self.previous_moves[depth] if depth < len(self.previous_moves) else None
            if move is None or not cands:
                return
            try:
                i = cands.index(move)
                if i != 0:
                    cands[0], cands[i] = cands[i], cands[0]
                    revs[0],  revs[i]  = revs[i],  revs[0]
            except ValueError:
                pass

        def max_value(chessboard, depth, alpha, beta) -> Tuple[float, Optional[Tuple[int, int]], bool, List[Tuple[int,int]]]:
            if time_exceeded():
                return self.evaluate(chessboard), None, True, []
            if depth == depth_limit:
                return self.evaluate(chessboard), None, False, []

            candidate_list, reversed_list = self.get_candidate_reversed_list(chessboard, COLOR_BLACK)
            reorder_with_previous(candidate_list, reversed_list, depth)
            # 启发式排序（保留 PV 在首位），其余按 |flips| 降序，再按权重降序
            if len(candidate_list) > 1:
                pairs = [(candidate_list[i], reversed_list[i]) for i in range(1, len(candidate_list))]
                pairs.sort(key=lambda x: (x[1].shape[0], int(self.weighted_board[x[0][0], x[0][1]])), reverse=True)
                candidate_list = [candidate_list[0]] + [p[0] for p in pairs]
                reversed_list  = [reversed_list[0]]  + [p[1] for p in pairs]
            if not candidate_list:  # 无合法步，检查是否终局，否则跳过回合
                opp_cands, _ = self.get_candidate_reversed_list(chessboard, COLOR_WHITE)
                if not opp_cands:
                    return self.evaluate(chessboard), None, False, []
                return min_value(chessboard, depth + 1, alpha, beta)

            best, move = -INFINITY, None
            best_pv: List[Tuple[int,int]] = []
            for candidate, reversed_opponents in zip(candidate_list, reversed_list):
                # 额外的频繁超时检查
                if time_exceeded():
                    # 若已有部分结果，直接返回当前最好解，以便迭代加深能利用
                    return (best if move is not None else self.evaluate(chessboard)), move, True, best_pv

                r0, c0 = candidate
                # 执行落子
                chessboard[r0, c0] = COLOR_BLACK
                k = reversed_opponents.shape[0]
                if k:
                    nb_flip_inplace(chessboard, reversed_opponents, COLOR_BLACK)

                v2, _, timed_out, child_pv = min_value(chessboard, depth + 1, alpha, beta)
                
                # 回退
                if k:
                    nb_flip_inplace(chessboard, reversed_opponents, COLOR_WHITE)
                chessboard[r0, c0] = COLOR_NONE

                if timed_out:
                    # 子节点已超时，直接把当前最好结果向上返回
                    return (best if move is not None else self.evaluate(chessboard)), move, True, best_pv

                if v2 > best:
                    best, move = v2, candidate
                    best_pv = [candidate] + child_pv
                # alpha-beta
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            return best, move, False, best_pv

        def min_value(chessboard, depth, alpha, beta) -> Tuple[float, Optional[Tuple[int, int]], bool, List[Tuple[int,int]]]:
            if time_exceeded():
                return self.evaluate(chessboard), None, True, []
            if depth == depth_limit:
                return self.evaluate(chessboard), None, False, []

            candidate_list, reversed_list = self.get_candidate_reversed_list(chessboard, COLOR_WHITE)
            reorder_with_previous(candidate_list, reversed_list, depth)
            # 启发式排序（保留 PV 在首位），其余按 |flips| 降序，再按权重降序
            if len(candidate_list) > 1:
                pairs = [(candidate_list[i], reversed_list[i]) for i in range(1, len(candidate_list))]
                pairs.sort(key=lambda x: (x[1].shape[0], int(self.weighted_board[x[0][0], x[0][1]])), reverse=True)
                candidate_list = [candidate_list[0]] + [p[0] for p in pairs]
                reversed_list  = [reversed_list[0]]  + [p[1] for p in pairs]
            if not candidate_list:  # 无合法步，检查是否终局，否则跳过回合
                opp_cands, _ = self.get_candidate_reversed_list(chessboard, COLOR_BLACK)
                if not opp_cands:
                    return self.evaluate(chessboard), None, False, []
                return max_value(chessboard, depth + 1, alpha, beta)

            best, move = INFINITY, None
            best_pv: List[Tuple[int,int]] = []
            for candidate, reversed_opponents in zip(candidate_list, reversed_list):
                if time_exceeded():
                    return (best if move is not None else self.evaluate(chessboard)), move, True, best_pv

                r0, c0 = candidate
                # 执行落子
                chessboard[r0, c0] = COLOR_WHITE
                k = reversed_opponents.shape[0]
                if k:
                    nb_flip_inplace(chessboard, reversed_opponents, COLOR_WHITE)

                v2, _, time_out, child_pv = max_value(chessboard, depth + 1, alpha, beta)

                # 回退
                if k:
                    nb_flip_inplace(chessboard, reversed_opponents, COLOR_BLACK)
                chessboard[r0, c0] = COLOR_NONE

                if time_out:
                    return (best if move is not None else self.evaluate(chessboard)), move, True, best_pv

                if v2 < best:
                    best, move = v2, candidate
                    best_pv = [candidate] + child_pv
                # alpha-beta
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break
            return best, move, False, best_pv

        if self.color == COLOR_BLACK:
            return max_value(chessboard, 0, -INFINITY, INFINITY) # 黑棋希望评估值越大越好
        else:
            return min_value(chessboard, 0, -INFINITY, INFINITY) # 白棋希望评估值越小越好

    def iterative_deepening(self, chessboard, start_time) -> Optional[Tuple[int,int]]:
        """
        迭代加深，在超时或达到上限即停止。
        优先返回已完成的最新深度结果。
        """
        best_move = None
        self.previous_moves: List[Optional[Tuple[int,int]]] = [None] * self.max_depth
        deadline = start_time + min(self.time_out, 4.9)

        for depth in range(5, self.max_depth + 1):
            # IDDFS 外层超时检查（留余量）
            if time.time() > deadline:
                break
            _, move, timed_out, best_pv = self.minimax_search(chessboard, depth, deadline)
            if timed_out:
                # 当前深度未完成，停止并使用上一轮的结果
                break
            if move is not None:
                best_move = move
                # 更新主变线提示，供下一轮各层排序使用
                self.previous_moves[:len(best_pv)] = best_pv
        return best_move
    
    def go(self, chessboard):
        start_time = time.time()
        self.candidate_list.clear()
        #============================================
        #Write your algorithm here
        chessboard = chessboard.astype(np.int8, copy=False)
        # 确认所有合法落子点
        self.candidate_list, _ = self.get_candidate_reversed_list(chessboard, self.color)

        if not self.candidate_list:         
            return self.candidate_list
        # 选择最终落子点
        best_move = self.iterative_deepening(chessboard, start_time)
        if best_move is not None:
            self.candidate_list.append(best_move)
        return self.candidate_list
