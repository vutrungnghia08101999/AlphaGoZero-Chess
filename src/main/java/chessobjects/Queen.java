package chessobjects;

import rules.Config;

public class Queen extends Piece {
    public Queen(int team) {
        super(team, Config.QUEEN_POINT);
    }

    @Override
    public String toString() {
        if(super.getTeam() == 0)
            return "QUEEN";
        return "queen";
    }
}
