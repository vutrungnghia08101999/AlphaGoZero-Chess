package rules;

import chessobjects.Board;
import chessobjects.Rook;
import utils.Spot;

import java.util.ArrayList;

public class RookRules extends AbstractRules {
    @Override
    public ArrayList<Spot> getInfluenceSpots(Spot start, Board board) {
        int row = start.getRow();
        int col = start.getCol();
        assert board.board[row][col] instanceof Rook;
        ArrayList<Spot> influenceSpots = new ArrayList<Spot>();
        for(int j = col + 1; j <= 8; ++j){
            influenceSpots.add(new Spot(row, j));
            if(board.board[row][j] != null)
                break;
        }
        for(int j = col - 1; j >= 1; --j){
            influenceSpots.add(new Spot(row, j));
            if(board.board[row][j] != null)
                break;
        }

        for(int i = row + 1; i <= 8; ++i){
            influenceSpots.add(new Spot(i, col));
            if(board.board[i][col] != null)
                break;
        }

        for(int i = row - 1; i >= 1; --i){
            influenceSpots.add(new Spot(i, col));
            if(board.board[i][col] != null)
                break;
        }

        return influenceSpots;
    }

    @Override
    public ArrayList<Spot> getReachedSpots(Spot start, Board board) {
        ArrayList<Spot> protectedSpots = getInfluenceSpots(start, board);
        ArrayList<Spot> reachedSpots = super.influenceToReachedSpots(start, protectedSpots, board);
        return reachedSpots;
    }
}
