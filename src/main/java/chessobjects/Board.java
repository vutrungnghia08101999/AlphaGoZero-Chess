package chessobjects;

public class Board {
    public Piece[][] board;
    public static int call = 0;

    public Board(){
        board = new Piece[9][9];

        board[1][1] = new Rook(0);
        board[1][2] = new Knight(0);
        board[1][3] = new Bishop(0);
        board[1][4] = new King(0);
        board[1][5] = new Queen(0);
        board[1][6] = new Bishop(0);
        board[1][7] = new Knight(0);
        board[1][8] = new Rook(0);
        for(int i = 1; i <= 8; ++i)
            board[2][i] = new Pawn(0);

        board[8][1] = new Rook(1);
        board[8][2] = new Knight(1);
        board[8][3] = new Bishop(1);
        board[8][4] = new King(1);
        board[8][5] = new Queen(1);
        board[8][6] = new Bishop(1);
        board[8][7] = new Knight(1);
        board[8][8] = new Rook(1);
        for(int i = 1; i <= 8; ++i)
            board[7][i] = new Pawn(1);
    }

    public void displayBoard(){
        System.out.println("\n\t\t\t  ____1________2________3________4________5________6________7________8____");
        for(int row = 1; row <= 8; ++row){
            System.out.println("\t\t\t  |        |        |        |        |        |        |        |        |");
            String s = String.format("\t\t\t%d |", row);
            for(int col = 1; col <= 8; ++col){
                if(board[row][col] == null)
                    s = s + "        |";
                else
                    s = s + String.format(" %-7s|", board[row][col].toString());
            }
            System.out.println(s);
            System.out.println("\t\t\t  |________|________|________|________|________|________|________|________|");
        }
        System.out.println();
    }


    public Piece getPiece(int row, int col){
        return this.board[row][col];
    }
    public Board clone(){
        Piece[][] tmp = new Piece[9][9];
        for(int row = 1; row <= 8; ++row){
            for(int col = 1; col <= 8; ++col){
                if(board[row][col] == null){
                    tmp[row][col] = null;
                }
                else if(board[row][col] instanceof King){
                    King king = (King)board[row][col];
                    tmp[row][col] = new King(king.getTeam());
                    ((King)tmp[row][col]).setCastlingPossible(king.isCastlingPossible());
                    ((King)tmp[row][col]).setCastlingDone(king.isCastlingDone());
                }
                else if(board[row][col] instanceof Queen){
                    Queen queen = (Queen)board[row][col];
                    tmp[row][col] = new Queen(queen.getTeam());
                }
                else if(board[row][col] instanceof Bishop){
                    Bishop bishop = (Bishop)board[row][col];
                    tmp[row][col] = new Bishop(bishop.getTeam());
                }
                else if(board[row][col] instanceof Knight){
                    Knight knight = (Knight)board[row][col];
                    tmp[row][col] = new Knight(knight.getTeam());
                }
                else if(board[row][col] instanceof Rook){
                    Rook rook = (Rook)board[row][col];
                    tmp[row][col] = new Rook(rook.getTeam());
                    ((Rook)tmp[row][col]).setCastlingPossible(rook.isCastlingPossible());
                }
                else if(board[row][col] instanceof Pawn){
                    Pawn pawn = (Pawn)board[row][col];
                    tmp[row][col] = new Pawn(pawn.getTeam());
                }
            }
        }

        Board boardTMP = new Board();
        boardTMP.board = tmp;
        return boardTMP;
    }
    private int[][] toMatrix() {
    	call++;
    	int[][] tmp = new int[9][9];
    	for(int row = 1; row <= 8; ++row){
            for(int col = 1; col <= 8; ++col){
                if(board[row][col] == null){
                    tmp[row][col] = 0;
                }
                else if(board[row][col] instanceof King){
                    tmp[row][col] = 1;
                    if (board[row][col].getTeam() == 1) 
                    	tmp[row][col] *= -1;
                }
                else if(board[row][col] instanceof Queen){
                    tmp[row][col] = 2;
                    if (board[row][col].getTeam() == 1) 
                    	tmp[row][col] *= -1;
                }
                else if(board[row][col] instanceof Bishop){
                    tmp[row][col] = 3;
                    if (board[row][col].getTeam() == 1) 
                    	tmp[row][col] *= -1;
                }
                else if(board[row][col] instanceof Knight){
                    tmp[row][col] = 4;
                    if (board[row][col].getTeam() == 1) 
                    	tmp[row][col] *= -1;
                }
                else if(board[row][col] instanceof Rook){
                    tmp[row][col] = 5;
                    if (board[row][col].getTeam() == 1) 
                    	tmp[row][col] *= -1;
                }
                else if(board[row][col] instanceof Pawn){
                    tmp[row][col] = 6;
                    if (board[row][col].getTeam() == 1) 
                    	tmp[row][col] *= -1;
                }
            }
        }
    	return tmp;
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
    	int[][] tmp = this.toMatrix();
    	int hash = 0;
    	for (int i = 1; i <= 8; i++) 
    		for (int j = 1; j <= 8; j++) {
    			if (tmp[i][j] != 0) {
    				long p = Board.powF((long) 2, (long) i + j);
        			p %= 1000000007;
        			p *= tmp[i][j];
        			p %= 1000000007;
        			hash +=  (int) p;
    			}
    			
    		}	
    	return hash;
    }
    @Override
    public boolean equals(Object b) {
    	Board board = (Board) b;
    	int[][] tmp2 = board.toMatrix();
    	int[][] tmp1 = this.toMatrix();
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
