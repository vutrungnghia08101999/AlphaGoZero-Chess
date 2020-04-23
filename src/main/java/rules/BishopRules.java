package rules;

import chessobjects.Bishop;
import chessobjects.Board;
import utils.Spot;

import java.util.ArrayList;

public class BishopRules extends AbstractRules {
    @Override
    public ArrayList<Spot> getInfluenceSpots(Spot start, Board board) {
        int row = start.getRow();
        int col = start.getCol();
        assert board.board[row][col] instanceof Bishop;
        ArrayList<Spot> influenceSpots = new ArrayList<Spot>();
        for(int i = 1; i <= 7; ++i){
            if(row + i > 8 || col + i > 8)
                break;
            influenceSpots.add(new Spot(row + i, col + i));
            if(board.board[row + i][col + i] != null)
                break;
        }
        for(int i = 1; i <= 7; ++i){
            if(row - i < 1 || col - i < 1)
                break;
            influenceSpots.add(new Spot(row - i, col - i));
            if(board.board[row - i][col - i] != null)
                break;
        }

        for(int i = 1; i <= 7; ++i){
            if(row - i < 1 || col + i > 8)
                break;
            influenceSpots.add(new Spot(row - i, col + i));
            if(board.board[row - i][col + i] != null)
                break;
        }

        for(int i = 1; i <= 7; ++i){
            if(row + i > 8 || col - i < 1)
                break;
            influenceSpots.add(new Spot(row + i, col - i));
            if(board.board[row + i][col - i] != null)
                break;
        }

        return influenceSpots;
    }

    @Override
    public ArrayList<Spot> getReachedSpots(Spot start, Board board) {
        ArrayList<Spot> influenceSpots = getInfluenceSpots(start, board);
        ArrayList<Spot> reachedSpots = super.influenceToReachedSpots(start, influenceSpots, board);
        return reachedSpots;
    }
}
