import numpy as np
from typing import Tuple, Optional, List

from agent import (
    COLOR_BLACK, COLOR_WHITE, COLOR_NONE,
    nb_get_possible_moves, nb_is_valid_move, nb_flip_inplace,
    nb_has_any_valid_move
)

class OthelloGame:
    """
    Othello game manager for tournament play between two agents.
    Supports both standard and reverse Othello rules.
    """
    
    def __init__(self, size: int = 8, reverse_mode: bool = True):
        """
        Initialize Othello game.
        
        Args:
            size: Board size (default 8x8)
            reverse_mode: True for reverse Othello (fewer pieces wins), False for standard
        """
        self.size = size
        self.reverse_mode = reverse_mode
        self.initial_state = self._create_initial_board()
    
    def _create_initial_board(self) -> np.ndarray:
        """Create standard Othello starting position."""
        board = np.zeros((self.size, self.size), dtype=np.int16)
        mid = self.size // 2
        # Standard starting position
        board[mid - 1, mid - 1] = COLOR_WHITE
        board[mid, mid] = COLOR_WHITE
        board[mid - 1, mid] = COLOR_BLACK
        board[mid, mid - 1] = COLOR_BLACK
        return board
    
    def get_legal_moves(self, board: np.ndarray, color: int) -> List[Tuple[int, int]]:
        """
        Get all legal moves for a player.
        
        Args:
            board: Current board state
            color: Player color (COLOR_BLACK or COLOR_WHITE)
            
        Returns:
            List of legal move coordinates
        """
        possible = nb_get_possible_moves(board, color)
        legal_moves = []
        for i in range(possible.shape[0]):
            r, c = int(possible[i, 0]), int(possible[i, 1])
            ok, flips = nb_is_valid_move(board, r, c, color)
            if ok and flips.shape[0] > 0:
                legal_moves.append((r, c))
        return legal_moves
    
    def apply_move(self, board: np.ndarray, move: Tuple[int, int], color: int) -> np.ndarray:
        """
        Apply a move and return new board state (non-destructive).
        
        Args:
            board: Current board state
            move: Move coordinates (row, col)
            color: Player color
            
        Returns:
            New board state after move
        """
        new_board = board.copy()
        r, c = move
        ok, flips = nb_is_valid_move(new_board, r, c, color)
        if not ok or flips.shape[0] == 0:
            raise ValueError(f"Invalid move: {move} for color {color}")
        
        new_board[r, c] = color
        nb_flip_inplace(new_board, flips, color)
        return new_board
    
    def is_terminal(self, board: np.ndarray) -> bool:
        """
        Check if game has ended (no legal moves for both players).
        
        Args:
            board: Current board state
            
        Returns:
            True if game is over
        """
        black_can_move = nb_has_any_valid_move(board, COLOR_BLACK)
        white_can_move = nb_has_any_valid_move(board, COLOR_WHITE)
        return not (black_can_move or white_can_move)
    
    def get_winner(self, board: np.ndarray) -> int:
        """
        Determine winner based on piece count.
        
        Args:
            board: Final board state
            
        Returns:
            COLOR_BLACK if black wins, COLOR_WHITE if white wins, COLOR_NONE for draw
        """
        black_count = int(np.sum(board == COLOR_BLACK))
        white_count = int(np.sum(board == COLOR_WHITE))
        
        if self.reverse_mode:
            # Reverse Othello: fewer pieces wins
            if black_count < white_count:
                return COLOR_BLACK
            elif white_count < black_count:
                return COLOR_WHITE
            else:
                return COLOR_NONE
        else:
            # Standard Othello: more pieces wins
            if black_count > white_count:
                return COLOR_BLACK
            elif white_count > black_count:
                return COLOR_WHITE
            else:
                return COLOR_NONE
    
    def get_score(self, board: np.ndarray) -> Tuple[int, int]:
        """
        Get current piece counts.
        
        Args:
            board: Current board state
            
        Returns:
            (black_count, white_count)
        """
        black_count = int(np.sum(board == COLOR_BLACK))
        white_count = int(np.sum(board == COLOR_WHITE))
        return black_count, white_count


def play_game(agent1, agent2, board_size: int = 8, reverse_mode: bool = True, 
              max_moves: int = 300, verbose: bool = False) -> dict:
    """
    Play a complete game between two agents.
    
    Args:
        agent1: First agent (plays as BLACK)
        agent2: Second agent (plays as WHITE)
        board_size: Size of the board
        reverse_mode: True for reverse Othello
        max_moves: Maximum number of moves before declaring draw
        verbose: Print game progress
        
    Returns:
        Dictionary with game result:
        {
            'winner': COLOR_BLACK/COLOR_WHITE/COLOR_NONE,
            'final_board': final board state,
            'black_score': black piece count,
            'white_score': white piece count,
            'num_moves': total moves played,
            'termination': 'normal'/'max_moves'/'error'
        }
    """
    game = OthelloGame(board_size, reverse_mode)
    board = game.initial_state.copy()
    
    agents = {COLOR_BLACK: agent1, COLOR_WHITE: agent2}
    current_color = COLOR_BLACK
    consecutive_passes = 0
    num_moves = 0
    
    try:
        while num_moves < max_moves:
            # Check if current player has legal moves
            legal_moves = game.get_legal_moves(board, current_color)
            
            if not legal_moves:
                # No legal moves - pass turn
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    # Both players passed - game over
                    break
                if verbose:
                    color_name = "BLACK" if current_color == COLOR_BLACK else "WHITE"
                    print(f"Move {num_moves}: {color_name} passes (no legal moves)")
                current_color = -current_color
                continue
            
            # Reset consecutive passes counter
            consecutive_passes = 0
            
            # Get agent's move
            agent = agents[current_color]
            candidate_list = agent.go(board.copy())
            
            if not candidate_list:
                # Agent returned no move despite having legal moves
                if verbose:
                    print(f"Agent returned empty candidate list")
                consecutive_passes += 1
                current_color = -current_color
                continue
            
            move = candidate_list[-1]  # Agent's decision is last in list
            
            # Validate and apply move
            if move not in legal_moves:
                raise ValueError(f"Invalid move {move} by agent")
            
            board = game.apply_move(board, move, current_color)
            num_moves += 1
            
            if verbose:
                color_name = "BLACK" if current_color == COLOR_BLACK else "WHITE"
                black_score, white_score = game.get_score(board)
                print(f"Move {num_moves}: {color_name} plays {move} | Score: B={black_score} W={white_score}")
            
            # Switch player
            current_color = -current_color
        
        # Determine winner
        winner = game.get_winner(board)
        black_score, white_score = game.get_score(board)
        
        termination = 'normal' if num_moves < max_moves else 'max_moves'
        
        if verbose:
            winner_name = "BLACK" if winner == COLOR_BLACK else "WHITE" if winner == COLOR_WHITE else "DRAW"
            print(f"\nGame Over - Winner: {winner_name}")
            print(f"Final Score: BLACK={black_score}, WHITE={white_score}")
            print(f"Total Moves: {num_moves}")
        
        return {
            'winner': winner,
            'final_board': board,
            'black_score': black_score,
            'white_score': white_score,
            'num_moves': num_moves,
            'termination': termination
        }
    
    except Exception as e:
        if verbose:
            print(f"Error during game: {e}")
        black_score, white_score = game.get_score(board)
        return {
            'winner': COLOR_NONE,
            'final_board': board,
            'black_score': black_score,
            'white_score': white_score,
            'num_moves': num_moves,
            'termination': 'error'
        }


def play_match(agent1, agent2, num_games: int = 10, board_size: int = 8, 
               reverse_mode: bool = True, verbose: bool = False) -> dict:
    """
    Play multiple games between two agents (alternating colors).
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        board_size: Size of the board
        reverse_mode: True for reverse Othello
        verbose: Print game progress
        
    Returns:
        Dictionary with match statistics:
        {
            'agent1_wins': number of wins as BLACK + as WHITE,
            'agent2_wins': number of wins as BLACK + as WHITE,
            'draws': number of draws,
            'agent1_as_black_wins': wins when playing as BLACK,
            'agent1_as_white_wins': wins when playing as WHITE,
            'total_games': total games played,
            'game_results': list of individual game results
        }
    """
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    agent1_as_black_wins = 0
    agent1_as_white_wins = 0
    game_results = []
    
    for i in range(num_games):
        # Alternate colors: even games agent1=BLACK, odd games agent1=WHITE
        if i % 2 == 0:
            result = play_game(agent1, agent2, board_size, reverse_mode, verbose=verbose)
            winner = result['winner']
            if winner == COLOR_BLACK:
                agent1_wins += 1
                agent1_as_black_wins += 1
            elif winner == COLOR_WHITE:
                agent2_wins += 1
            else:
                draws += 1
        else:
            result = play_game(agent2, agent1, board_size, reverse_mode, verbose=verbose)
            winner = result['winner']
            if winner == COLOR_BLACK:
                agent2_wins += 1
            elif winner == COLOR_WHITE:
                agent1_wins += 1
                agent1_as_white_wins += 1
            else:
                draws += 1
        
        game_results.append(result)
        
        if verbose:
            print(f"Game {i+1}/{num_games} complete\n")
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'agent1_as_black_wins': agent1_as_black_wins,
        'agent1_as_white_wins': agent1_as_white_wins,
        'total_games': num_games,
        'game_results': game_results
    }
