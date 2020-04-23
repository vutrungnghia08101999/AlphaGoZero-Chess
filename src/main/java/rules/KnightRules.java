package rules;

import chessobjects.Board;
import chessobjects.Knight;
import utils.Spot;

import java.util.ArrayList;

public class KnightRules extends AbstractRules {
    @Override
    public ArrayList<Spot> getInfluenceSpots(Spot start, Board board) {
        int row = start.getRow();
        int col = start.getCol();
        assert board.board[row][col] instanceof Knight;
        ArrayList<Spot> influenceSpots = new ArrayList<Spot>();
        if(row - 2 >= 1 && col + 1 <= 8)
            influenceSpots.add(new Spot(row - 2, col + 1));
        if(row - 1 >= 1 && col + 2 <= 8)
            influenceSpots.add(new Spot(row - 1, col + 2));

        if(row + 1 <= 8 && col + 2 <= 8)
            influenceSpots.add(new Spot(row + 1, col + 2));
        if(row + 2 <= 8 && col + 1 <= 8)
            influenceSpots.add(new Spot(row + 2, col + 1));

        if(row + 2 <= 8 && col - 1 >= 1)
            influenceSpots.add(new Spot(row + 2, col - 1));
        if(row + 1 <= 8 && col - 2 >= 1)
            influenceSpots.add(new Spot(row + 1, col - 2));

        if(row - 1 >= 1 && col - 2 >= 1)
            influenceSpots.add(new Spot(row - 1, col - 2));
        if(row - 2 >= 1 && col - 1 >= 1)
            influenceSpots.add(new Spot(row - 2, col - 1));

        return influenceSpots;
    }

    @Override
    public ArrayList<Spot> getReachedSpots(Spot start, Board board) {
        ArrayList<Spot> protectedSpots = getInfluenceSpots(start, board);
        ArrayList<Spot> influenceSpots = super.influenceToReachedSpots(start, protectedSpots, board);
        return influenceSpots;
    }
}
