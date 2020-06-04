package engine;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import chessobjects.Board;
import chessobjects.King;
import chessobjects.Pawn;
import chessobjects.Piece;
import chessobjects.Queen;
import chessobjects.Rook;
import rules.AbstractRules;
import sun.nio.ch.Util;
import ttable.Config;
import ttable.LRUCache;
import ttable.NodeTT;
import ttable.Value;
import utils.Move;
import utils.Spot;

public class CPU {
	private int alphaBetaCall, negaScoutCall, alphaBetaTTCall, TTCall, getAlphaBetaCall;
	private Move nextMoves;
	private ArrayList<Move> history = new ArrayList<Move>();

	private LRUCache<Board, Value> team = new LRUCache<Board, Value>(Config.CAP);

	public CPU() {
		this.alphaBetaCall = 0;
		this.negaScoutCall = 0;
		this.alphaBetaTTCall = 0;
		this.TTCall = 0;
		this.team.map = new HashMap<Board, NodeTT<Board, Value>>();
	}

	public boolean isValidMove(Move playerMove, Board board, int team) {
		assert playerMove != null;
		assert playerMove.getStart() != null;
		assert playerMove.getEnd() != null;

		ArrayList<Move> allValidMoves = this.getAllValidMoves(board, team);
		return Utils.isContainedMove(allValidMoves, playerMove);
	}

	public boolean isCheckedMate(Board board, int team) {
		ArrayList<Move> allValidMoves = this.getAllValidMoves(board, team);
		return allValidMoves.size() == 0;
	}

	public Move searchNextMove(Board board, int team, int ROOT_TREE_DEPTH) {
		this.nextMoves = null;
		this.alphaBetaCall = 0;
		negaScoutCall = 0;
		this.TTCall = 0;
		this.alphaBetaTTCall = 0;
		Board.call = 0;
		int maximum;
//        	maximum = this.dfsAlphaBeta(board, team, ROOT_TREE_DEPTH, team, ROOT_TREE_DEPTH, -100000000, 100000000);
//        	System.out.println("Alpha Beta: " + this.alphaBetaCall);
//		int firstG = this.iterativeDeepening(board, ROOT_TREE_DEPTH, team);
//		if (team == 0)
//			maximum = this.MTDf(board, 0, ROOT_TREE_DEPTH + 1, team);
//		else
		maximum = this.dfsAlphaBeta(board, team, ROOT_TREE_DEPTH, team, ROOT_TREE_DEPTH, -100000000, 100000000);
//        	System.out.println("First Guess Values: " + firstG);
//        	System.out.println("Rate of TT: " + this.TTCall * 1.0 / this.alphaBetaTTCall + " --- " + this.team.map.size() + " --- " + this.alphaBetaTTCall);
		System.out.println("Minimize-maximize algorithm metrics: " + maximum);
//		System.out.println("Number cal Matrix: " + Board.call);
		System.out.println("Number of DFS Calls: " + this.alphaBetaCall);
		ArrayList<Move> tmp = new ArrayList<Move>();
		for (int i = Math.max(0, history.size() - 9); i < history.size(); ++i)  // avoid repeated moves
			tmp.add(history.get(i));
		history = tmp;
		history.add(nextMoves);
		return nextMoves;
	}

	public Board getNextState(Board board, Move move) {
		Board B = board.clone();
		Spot start = move.getStart();
		Spot end = move.getEnd();

		int start_row = start.getRow();
		int start_col = start.getCol();
		int end_row = end.getRow();
		int end_col = end.getCol();

		int team = B.board[start_row][start_col].getTeam();

		/********************** Update castling state *************/
		if (B.board[start_row][start_col] instanceof King)
			((King) B.board[start_row][start_col]).setCastlingPossible(false);
		else if (B.board[start_row][start_col] instanceof Rook)
			((Rook) B.board[start_row][start_col]).setCastlingPossible(false);

		/***********
		 * Update board refer to 3 case: castling, promoted, normal
		 **********/
		if (move.isCastling()) {
			assert B.board[start_row][start_col] instanceof King;
			B.board[end_row][end_col] = B.board[start_row][start_col];
			B.board[start_row][start_col] = null;
			((King) B.board[end_row][end_col]).setCastlingDone(true);

			B.digitBoard[end_row][end_col] = B.digitBoard[start_row][start_col];
			B.digitBoard[start_row][start_col] = 0;
			if (end_col == 3) {
				B.board[end_row][4] = B.board[end_row][1];
				B.board[end_row][1] = null;
				((Rook) B.board[end_row][4]).setCastlingPossible(false);

				B.digitBoard[end_row][4] = B.digitBoard[end_row][1];
				B.digitBoard[end_row][1] = 0;
			} else if (end_col == 7) {
				B.board[end_row][6] = B.board[end_row][8];
				B.board[end_row][8] = null;
				((Rook) B.board[end_row][6]).setCastlingPossible(false);

				B.digitBoard[end_row][6] = B.digitBoard[end_row][8];
				B.digitBoard[end_row][8] = 0;
			}

		} else if (move.isPromoted()) {
			B.board[start_row][start_col] = null;
			B.board[end_row][end_col] = new Queen(team);

			B.digitBoard[start_row][start_col] = 0;
			B.digitBoard[end_row][end_col] = B.QUEEN + team * 6;
		} else {
			B.board[end_row][end_col] = B.board[start_row][start_col];
			B.board[start_row][start_col] = null;

			B.digitBoard[end_row][end_col] = B.digitBoard[start_row][start_col];
			B.digitBoard[start_row][start_col] = 0;
		}
		return B;
	}

	/*********************** DFS FUNCTIONS *********************************/

	private int dfsAlphaBeta(Board board, int flag, int depth, int team, int ROOT_TREE_DEPTH, int alpha, int beta) {
		this.alphaBetaCall++;
		if (depth == 1)
			return this.evaluate(board, team);

		ArrayList<Move> moves = this.getAllValidMoves(board, flag);
		if (moves.size() == 0){
			if (flag == team)
				return -(1000 + depth);
			else
				return 1000 + depth;
		}

		int nextTeam = Math.abs(1 - flag);
		if (flag == team) {
			int maximum = -100000000;
			for (Move move : moves) {
				Board B = this.getNextState(board, move);
				int metric = dfsAlphaBeta(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, alpha, beta);
				if (depth == ROOT_TREE_DEPTH && Utils.isContainedMove(this.history, move) && moves.size() != 1)
					metric = -100000000;
				if (metric > maximum) {
					maximum = metric;
					if (depth == ROOT_TREE_DEPTH)
						nextMoves = move;
				}
				alpha = Math.max(metric, alpha);
				if (beta <= alpha)
					break;
			}
			return maximum;
		} else {
			int minimum = 100000000;
			for (Move move : moves) {
				Board B = this.getNextState(board, move);
				int metric = dfsAlphaBeta(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, alpha, beta);
				minimum = Math.min(minimum, metric);
				beta = Math.min(beta, metric);
				if (beta <= alpha)
					break;
			}
			return minimum;
		}
	}

	private int negaScout(Board board, int flag, int depth, int team, int ROOT_TREE_DEPTH, int alpha, int beta) {
		this.negaScoutCall++;
		if (depth == 1) {
			return this.evaluate(board, team);
		}
		ArrayList<Move> moves = this.getAllValidMoves(board, flag);
		if (moves.size() == 0)
			return this.evaluate(board, team);
		int nextTeam = Math.abs(1 - flag);

		if (flag == team) {
			int m = -100000000;
			Board B = this.getNextState(board, moves.get(0));
			m = Math.max(m, negaScout(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, alpha, beta));

			if (depth == ROOT_TREE_DEPTH)
				nextMoves = moves.get(0);
			if (m >= beta) {
				if (depth == ROOT_TREE_DEPTH) {
					return m;
				}

			}

			for (int i = 1; i < moves.size(); i++) {
				B = this.getNextState(board, moves.get(i));
				int t = negaScout(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, m, m + 1);
				if (t > m) {
					if (depth == ROOT_TREE_DEPTH)
						nextMoves = moves.get(i);
					if (depth < 3 || t >= beta) {
						m = t;
					} else {
						m = negaScout(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, t, beta);
					}
				}
				if (m >= beta) {
					if (depth == ROOT_TREE_DEPTH) {
						nextMoves = moves.get(i);
						return m;
					}

				}

			}
			return m;
		} else {
			int m = 100000000;
			Board B = this.getNextState(board, moves.get(0));
			m = Math.min(m, negaScout(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, alpha, beta));

			if (m <= alpha)
				return m;
			for (int i = 1; i < moves.size(); i++) {
				B = this.getNextState(board, moves.get(i));
				int t = negaScout(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, m - 1, m);

				if (t < m) {
					if (depth < 3 || t <= alpha) {
						m = t;
					} else {
						m = negaScout(B, nextTeam, depth - 1, team, ROOT_TREE_DEPTH, alpha, t);
					}
				}
				if (m <= alpha)
					return m;
			}
			return m;
		}

	}

	private int alphaBetaTT(Board board, int flag, int depth, int team, int alpha, int beta) {
		this.alphaBetaTTCall++;
		int value;
		Value v = this.team.get(board);
		if (depth == 1) {
//			if (v != null && v.depth == depth)
//				return v.lower;
			int val = this.evaluate(board, team);
    		this.team.put(board, new Value(depth, val, val));
			return val;
		}
		
		if (v != null && v.depth >= depth) {
			this.TTCall += 1;
			if (v.lower >= beta)
				return v.lower;
			if (v.upper <= alpha)
				return v.upper;
			alpha = Math.max(alpha, v.lower);
			beta = Math.min(beta, v.upper);
		}

		
		
		ArrayList<Move> moves = this.getAllValidMoves(board, flag);

		if (moves.size() == 0) {
//			if (v != null && v.depth == depth)
//				return v.lower;
			int val = this.evaluate(board, team);
    		this.team.put(board, new Value(depth, val, val));
			return val;
		}
		
		int g;
		int nextTeam = 1 - flag;

		if (flag == team) {
			g = Integer.MIN_VALUE;
			int a = alpha;
			for (Move move : moves) {
				Board B = this.getNextState(board, move);
				int g_tmp = this.alphaBetaTT(B, nextTeam, depth - 1, team, a, beta);
				if (g < g_tmp) {
					if (depth == rules.Config.TREE_DEPTH + 1)
						this.nextMoves = move;
					g = g_tmp;
				}
				a = Math.max(a, g);
				if (g >= beta)
					break;
			}
		} else {
			g = Integer.MAX_VALUE;
			int b = beta;
			for (Move move : moves) {
				Board B = this.getNextState(board, move);
//    			System.out.println("Hash COde:= " + B.hashCode());
				g = Math.min(g, this.alphaBetaTT(B, nextTeam, depth - 1, team, alpha, b));
				b = Math.min(g, b);
				if (g <= alpha) {
					break;
				}
			}
		}

		Value newV = new Value(depth);
		if (g <= alpha) {
			newV.upper = g;
			this.team.put(board, newV);
		} else if (g < beta) {
			newV.lower = g;
			newV.upper = g;
			this.team.put(board, newV);
		} else {
			newV.lower = g;
			this.team.put(board, newV);
		}

		return g;

	}

	public int MTDf(Board board, int first, int depth, int team) {
		int g = first;
		int beta;
		int lowerB = -100000000, upperB = 100000000;
		do {

			long t1 = System.currentTimeMillis();
			this.alphaBetaTTCall = 0;
			this.TTCall = 0;
			System.out.println("Lower:= " + lowerB + " ----- " + "Upper:= " + upperB);

//    		beta = (lowerB + upperB + 1) / 2;
			if (g == lowerB)
				beta = g + 1;
			else
				beta = g;
			g = this.alphaBetaTT(board, team, depth, team, beta - 1, beta);
    		System.out.println("Lower:= " + lowerB + " ----- " + "Upper:= " + upperB + " --- Returned:= " + g + " --- Beta:= " + beta);
			if (g < beta) {
				upperB = g;
			} else {
				lowerB = g;
			}

			long t2 = System.currentTimeMillis();
			System.out.println(" --- Time for one loop:= " + (t2 - t1));
			System.out.println("Rate of TT: " + this.TTCall * 1.0 / this.alphaBetaTTCall + " --- "
					+ this.team.map.size() + " --- " + this.alphaBetaTTCall);
			System.out.println(" **-------------------------------------------** ");

		} while (lowerB < upperB);
		return g;
	}

	private int iterativeDeepening(Board board, int depth, int team) {
		long t = 1500;
		System.out.println("... Time Start For Iterative Deepening ...");
		long beg = System.currentTimeMillis();
		int first = 0;
		for (int d = 1; d <= depth; d++) {
			first = this.MTDf(board, first, d, team);
			long t2 = System.currentTimeMillis();
			if (t2 - beg > t)
				break;
		}
		System.out.println("... Time Out For Iterative Deepening ...");
		return first;
	}

	/************************ SUPPORTED FUNCTIONS **************************/

	private ArrayList<Move> getAllValidMoves(Board board, int team) {
		ArrayList<Move> allValidMoves = new ArrayList<Move>();
		for (int row = 1; row <= 8; ++row) {
			for (int col = 1; col <= 8; ++col) {
				if (board.board[row][col] == null)
					continue;
				if (board.board[row][col].getTeam() != team)
					continue;
				Spot start = new Spot(row, col);
				ArrayList<Move> validSpots = getValidMoves(start, board);
				allValidMoves.addAll(validSpots);
			}
		}

//        Collections.shuffle(allValidMoves);
		return allValidMoves;
	}

	private ArrayList<Move> getValidMoves(Spot start, Board board) {
		int row = start.getRow();
		int col = start.getCol();
		assert board.board[row][col] != null;
		int team = board.board[row][col].getTeam();
		Piece piece = board.board[row][col];

		AbstractRules engine = Utils.chooseRules(board.board[row][col]);
		ArrayList<Spot> reachedSpots = engine.getReachedSpots(start, board);
		ArrayList<Move> validMoves = new ArrayList<Move>();
		for (Spot spot : reachedSpots) {
			boolean isPromotedMove = (piece instanceof Pawn)
					&& ((team == 0 && spot.getRow() == 8) || (team == 1 && spot.getRow() == 1));
			if (isPromotedMove) {
				Move move = new Move(start, spot, false, true);
				Board B = this.getNextState(board, move);
				if (!this.isChecked(B, team))
					validMoves.add(move);
			} else {
				Move move = new Move(start, spot);
				Board B = this.getNextState(board, move);
				if (!this.isChecked(B, team))
					validMoves.add(move);
			}
		}

		if (board.board[row][col] instanceof King) {
			ArrayList<Move> castlingSpots = this.getCastlingMoves(start, board);
			validMoves.addAll(castlingSpots);
		}

//        Collections.shuffle(validMoves);
		return validMoves;
	}

	public boolean isChecked(Board board, int team) {
		Spot kingPosition = Utils.getKingPosition(board, team);
		assert kingPosition.getRow() != -1;
		assert kingPosition.getCol() != -1;

		AbstractRules engine;
		for (int row = 1; row <= 8; ++row) {
			for (int col = 1; col <= 8; ++col) {
				if (board.board[row][col] == null)
					continue;
				if (board.board[row][col].getTeam() == team)
					continue;

				engine = Utils.chooseRules(board.board[row][col]);
				ArrayList<Spot> influenceSpots = engine.getInfluenceSpots(new Spot(row, col), board);
				if (Utils.isContainedSpot(influenceSpots, kingPosition)) // ERROR: Spot can contain isCastlingPossible
																			// and isCastlingDone => this is really bad.
					return true;
			}
		}
		return false;
	}

	private ArrayList<Move> getCastlingMoves(Spot start, Board board) {
		int row = start.getRow();
		int col = start.getCol();
		assert board.board[row][col] instanceof King;
		ArrayList<Move> castlingMoves = new ArrayList<Move>();

		King king = (King) board.board[row][col];
		int team = king.getTeam();

		if (this.isChecked(board, team))
			return castlingMoves;

		if (king.isCastlingPossible()) {
			assert col == 5;
			// right 0-0
			if (board.board[row][8] instanceof Rook && board.board[row][6] == null && board.board[row][7] == null) {
				Rook rook = (Rook) board.board[row][8];
				if (rook.isCastlingPossible()) {
					Board B1 = board.clone();
					B1.board[row][6] = B1.board[row][col];
					B1.board[row][col] = null;
					if (!this.isChecked(B1, team)) {
						Board B2 = board.clone();
						B2.board[row][7] = B2.board[row][col];
						B2.board[row][col] = null;
						if (!this.isChecked(B2, team)) {
							Move move = new Move(start, new Spot(row, 7), true, false);
							castlingMoves.add(move);
						}
					}
				}
			}
			if (board.board[row][1] instanceof Rook && board.board[row][2] == null && board.board[row][3] == null
					&& board.board[row][4] == null) {
				Rook rook = (Rook) board.board[row][1];
				if (rook.isCastlingPossible()) {
					Board B1 = board.clone();
					B1.board[row][4] = B1.board[row][col];
					B1.board[row][col] = null;
					if (!this.isChecked(B1, team)) {
						Board B2 = board.clone();
						B2.board[row][3] = B2.board[row][col];
						B2.board[row][col] = null;
						if (!this.isChecked(B2, team)) {
							Move move = new Move(start, new Spot(row, 3), true, false);
							castlingMoves.add(move);
						}
					}
				}
			}
		}
		return castlingMoves;
	}

	/***************************
	 * OBJECTIVES **************************************
	 * https://chessfox.com/free-chess-course-chessfox-com/introduction-to-the-5-main-objectives-of-a-chess-game/
	 */
	public int evaluate(Board board, int team) {
//		if (this.isCheckedMate(board, team))
//			return -1000;
//		else if (this.isCheckedMate(board, 1 - team))
//			return 1000;
		int materialScore = this.getMaterialScore(board, team);
//        float developmentScore = this.getDevelopmentScore(board, team);
//        float centerControlScore = this.getCenterControlScore(board, team);
//		int kingSafetyScore = this.getKingSafetyScore(board, team);
//        float pawnStructureScore = this.getPawnStructureScore(board, team);
		return materialScore;
	}

	private int getMaterialScore(Board board, int team) {
		int ourPoint = 0;
		int opponentPoint = 0;
		for (int row = 1; row <= 8; ++row) {
			for (int col = 1; col <= 8; ++col) {
				if (board.board[row][col] == null)
					continue;

				Piece piece = board.board[row][col];
				if (board.board[row][col].getTeam() != team) {
					if (piece instanceof Pawn)
						opponentPoint += ((Pawn) piece).getPoint(row);
					else
						opponentPoint += piece.getPoint();
				} else {
					if (piece instanceof Pawn)
						ourPoint += ((Pawn) piece).getPoint(row);
					else
						ourPoint += piece.getPoint();
				}
			}
		}
		return ourPoint - opponentPoint;
	}

	private float getCenterControlScore(Board board, int team) {
		int MAXIMUM_VALUE = 10; // 4 pawns + 4 other pieces for influence and 2 for occupied, this is my
								// assumption
		int ourPieces = 0;
		int opponentPieces = 0;

		for (int row = 1; row <= 8; ++row) {
			for (int col = 1; col <= 8; ++col) {
				Piece piece = board.board[row][col];
				if (piece == null)
					continue;
				if (piece instanceof King)
					continue;

				if (Utils.isInCenter(new Spot(row, col))) {
					if (piece.getTeam() == team)
						ourPieces++;
					else
						opponentPieces++;
				}

				AbstractRules rules = Utils.chooseRules(piece);
				ArrayList<Spot> influenceSpots = rules.getInfluenceSpots(new Spot(row, col), board);
				for (Spot spot : influenceSpots) {
					if (Utils.isInCenter(spot)) {
						if (piece.getTeam() == team)
							ourPieces++;
						else
							opponentPieces++;
						break;
					}
				}
			}
		}
		return ((float) (ourPieces - opponentPieces)) / MAXIMUM_VALUE;
	}

	private int getKingSafetyScore(Board board, int team) {
		int ourKingSafetyScore = 0;
		int opponentKingSafetyScore = 0;

		int opponent = Math.abs(1 - team);
		Spot ourKingPos = Utils.getKingPosition(board, team);
		Spot opponentKingPos = Utils.getKingPosition(board, opponent);
		King ourKing = (King) board.board[ourKingPos.getRow()][ourKingPos.getCol()];
		King opponentKing = (King) board.board[opponentKingPos.getRow()][opponentKingPos.getCol()];
		// castling score
		if (ourKing.isCastlingDone())
			ourKingSafetyScore += 2;
		if (opponentKing.isCastlingDone())
			opponentKingSafetyScore += 2;
		// pawn score
		int ourKingStartCol = Math.max(ourKingPos.getCol() - 1, 1);
		int ourKingEndCol = Math.min(ourKingPos.getCol() + 1, 8);
		int opponentKingStartCol = Math.max(opponentKingPos.getCol() - 1, 1);
		int opponentKingEndCol = Math.min(opponentKingPos.getCol() + 1, 8);
		if (team == 0) {
			if (ourKingPos.getRow() < 8) {
				for (int col = ourKingStartCol; col <= ourKingEndCol; ++col) {
					Piece piece = board.board[ourKingPos.getRow() + 1][col];
					if (piece instanceof Pawn) {
						if (piece.getTeam() == team)
							ourKingSafetyScore++;
					}
				}
			}
			if (opponentKingPos.getRow() > 1) {
				for (int col = opponentKingStartCol; col <= opponentKingEndCol; ++col) {
					Piece piece = board.board[opponentKingPos.getRow() - 1][col];
					if (piece instanceof Pawn) {
						if (piece.getTeam() == opponent)
							opponentKingSafetyScore++;
					}
				}
			}
		} else {
			if (ourKingPos.getRow() > 1) {
				for (int col = ourKingStartCol; col <= ourKingEndCol; ++col) {
					Piece piece = board.board[ourKingPos.getRow() - 1][col];
					if (piece instanceof Pawn) {
						if (piece.getTeam() == team)
							ourKingSafetyScore++;
					}
				}
			}
			if (opponentKingPos.getRow() < 8) {
				for (int col = opponentKingStartCol; col <= opponentKingEndCol; ++col) {
					Piece piece = board.board[opponentKingPos.getRow() + 1][col];
					if (piece instanceof Pawn) {
						if (piece.getTeam() == opponent)
							opponentKingSafetyScore++;
					}
				}
			}
		}
		// defender and threaten
		for (int row = 1; row <= 8; ++row) {
			for (int col = 1; col <= 8; ++col) {
				Piece piece = board.board[row][col];
				if (piece == null)
					continue;
				if (piece instanceof King)
					continue;
				AbstractRules rules = Utils.chooseRules(piece);
				ArrayList<Spot> influenceSpots = rules.getInfluenceSpots(new Spot(row, col), board);
				if (piece.getTeam() == team) {
					for (Spot spot : influenceSpots) {
						if (Utils.isAdjacentOrTheSame(ourKingPos, spot) && !(piece instanceof Pawn)) {
							ourKingSafetyScore++;
							break;
						}
					}
					for (Spot spot : influenceSpots) {
						if (Utils.isAdjacentOrTheSame(opponentKingPos, spot)) {
							opponentKingSafetyScore--;
							break;
						}
					}
				} else {
					for (Spot spot : influenceSpots) {
						if (Utils.isAdjacentOrTheSame(ourKingPos, spot)) {
							ourKingSafetyScore--;
							break;
						}
					}
					for (Spot spot : influenceSpots) {
						if (Utils.isAdjacentOrTheSame(opponentKingPos, spot) && !(piece instanceof Pawn)) {
							opponentKingSafetyScore++;
							break;
						}
					}
				}
			}
		}

		return ourKingSafetyScore - opponentKingSafetyScore;
	}

	private float getPawnStructureScore(Board board, int team) {
		int MAXIMUM_VALUE = 8; // a weak point if it doesn't connect to other pawn or there are two pawn in a
								// column
		int ourWeakPawns = 0;
		int opponentWeakPawns = 0;

		for (int row = 1; row <= 8; ++row) {
			for (int col = 1; col <= 8; ++col) {
				Piece piece = board.board[row][col];
				if (piece == null)
					continue;
				if (piece instanceof Pawn) {
					if (Utils.isWeakPawn(new Spot(row, col), board)) {
						if (piece.getTeam() == team)
							ourWeakPawns++;
						else
							opponentWeakPawns++;
					}
				}
			}
		}

		return (-1) * ((float) (ourWeakPawns - opponentWeakPawns)) / MAXIMUM_VALUE;
	}
}
