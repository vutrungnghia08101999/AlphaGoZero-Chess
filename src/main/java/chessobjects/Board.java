package chessobjects;

public class Board {
	public Piece[][] board;
	public int[][] digitBoard;
	public int KING = 1;
	public int QUEEN = 2;
	public int ROOK = 3;
	public int BISHOP = 4;
	public int KNIGHT = 5;
	public int PAWN = 6;
	public static int call = 0;

	public Board() {
		board = new Piece[9][9];

		board[1][1] = new Rook(0);
		board[1][2] = new Knight(0);
		board[1][3] = new Bishop(0);
		board[1][4] = new Queen(0);
		board[1][5] = new King(0);
		board[1][6] = new Bishop(0);
		board[1][7] = new Knight(0);
		board[1][8] = new Rook(0);
		for (int i = 1; i <= 8; ++i)
			board[2][i] = new Pawn(0);

		board[8][1] = new Rook(1);
		board[8][2] = new Knight(1);
		board[8][3] = new Bishop(1);
		board[8][4] = new Queen(1);
		board[8][5] = new King(1);
		board[8][6] = new Bishop(1);
		board[8][7] = new Knight(1);
		board[8][8] = new Rook(1);
		for (int i = 1; i <= 8; ++i)
			board[7][i] = new Pawn(1);

		digitBoard = new int[9][9];
		digitBoard[1][1] = ROOK;
		digitBoard[1][2] = KNIGHT;
		digitBoard[1][3] = BISHOP;
		digitBoard[1][4] = QUEEN;
		digitBoard[1][5] = KING;
		digitBoard[1][6] = BISHOP;
		digitBoard[1][7] = KNIGHT;
		digitBoard[1][8] = ROOK;
		for (int i = 1; i <= 8; ++i)
			digitBoard[2][i] = PAWN;

		digitBoard[8][1] = ROOK + 6;
		digitBoard[8][2] = KNIGHT + 6;
		digitBoard[8][3] = BISHOP + 6;
		digitBoard[8][4] = QUEEN + 6;
		digitBoard[8][5] = KING + 6;
		digitBoard[8][6] = BISHOP + 6;
		digitBoard[8][7] = KNIGHT + 6;
		digitBoard[8][8] = ROOK + 6;
		for (int i = 1; i <= 8; ++i)
			digitBoard[7][i] = PAWN + 6;

		for (int row = 3; row <= 6; ++row)
			for (int col = 1; col <= 8; ++col)
				digitBoard[row][col] = 0;
	}

	public void displayBoard() {
		System.out.println("\n\t\t\t  ____1________2________3________4________5________6________7________8____");
		for (int row = 1; row <= 8; ++row) {
			System.out.println("\t\t\t  |        |        |        |        |        |        |        |        |");
			String s = String.format("\t\t\t%d |", row);
			for (int col = 1; col <= 8; ++col) {
				if (board[row][col] == null)
					s = s + "        |";
				else
					s = s + String.format(" %-7s|", board[row][col].toString());
			}
			System.out.println(s);
			System.out.println("\t\t\t  |________|________|________|________|________|________|________|________|");
		}
		System.out.println();
//		System.out.println("\n\t\t\t  ____1________2________3________4________5________6________7________8____");
//		for (int row = 1; row <= 8; ++row) {
//			System.out.println("\t\t\t  |        |        |        |        |        |        |        |        |");
//			String s = String.format("\t\t\t%d |", row);
//			for (int col = 1; col <= 8; ++col) {
//				if (board[row][col] == null)
////					s = s + "        |";
//					s = s + String.format(" %-7d|", digitBoard[row][col]);
//				else
//					s = s + String.format(" %-7d|", digitBoard[row][col]);
//			}
//			System.out.println(s);
//			System.out.println("\t\t\t  |________|________|________|________|________|________|________|________|");
//		}
//		System.out.println();

	}

	public Piece getPiece(int row, int col) {
		return this.board[row][col];
	}

	public Board clone() {
		Piece[][] tmp = new Piece[9][9];
		for (int row = 1; row <= 8; ++row) {
			for (int col = 1; col <= 8; ++col) {
				if (board[row][col] == null) {
					tmp[row][col] = null;
				} else if (board[row][col] instanceof King) {
					King king = (King) board[row][col];
					tmp[row][col] = new King(king.getTeam());
					((King) tmp[row][col]).setCastlingPossible(king.isCastlingPossible());
					((King) tmp[row][col]).setCastlingDone(king.isCastlingDone());
				} else if (board[row][col] instanceof Queen) {
					Queen queen = (Queen) board[row][col];
					tmp[row][col] = new Queen(queen.getTeam());
				} else if (board[row][col] instanceof Bishop) {
					Bishop bishop = (Bishop) board[row][col];
					tmp[row][col] = new Bishop(bishop.getTeam());
				} else if (board[row][col] instanceof Knight) {
					Knight knight = (Knight) board[row][col];
					tmp[row][col] = new Knight(knight.getTeam());
				} else if (board[row][col] instanceof Rook) {
					Rook rook = (Rook) board[row][col];
					tmp[row][col] = new Rook(rook.getTeam());
					((Rook) tmp[row][col]).setCastlingPossible(rook.isCastlingPossible());
				} else if (board[row][col] instanceof Pawn) {
					Pawn pawn = (Pawn) board[row][col];
					tmp[row][col] = new Pawn(pawn.getTeam());
				}
			}
		}

		int[][] tmpDigit = new int[9][9];
		for (int row = 1; row <= 8; ++row)
			for (int col = 1; col <= 8; ++col)
				tmpDigit[row][col] = digitBoard[row][col];

		Board boardTMP = new Board();
		boardTMP.board = tmp;
		boardTMP.digitBoard = tmpDigit;
		return boardTMP;
	}


	public static long powF(long a, long b) {
		long re = 1;
		while (b > 0) {
			if ((b & 1) == 1) {
				re *= a;
			}
			b >>= 1;
			a *= a;
		}
		return re;
	}

	@Override
	public int hashCode() {
		call++;
		int[][] tmp = this.digitBoard;
		int hash = 0;
		for (int i = 1; i <= 8; i++)
			for (int j = 1; j <= 8; j++) {
				if (tmp[i][j] != 0) {
					long p = Board.powF((long) 2, (long) i * j - 2);
					p *= tmp[i][j];
					hash += (int)(p ^ (p >>> 32));
				}

			}
		return hash;
	}

	@Override
	public boolean equals(Object b) {
		call++;
		Board board = (Board) b;
		int[][] tmp2 = board.digitBoard;
		int[][] tmp1 = this.digitBoard;
		for (int i = 1; i <= 8; i++) {
			for (int j = 1; j <= 8; j++) {
				if (tmp1[i][j] != tmp2[i][j]) {
					return false;
				}
			}
		}
		return true;
	}
}
