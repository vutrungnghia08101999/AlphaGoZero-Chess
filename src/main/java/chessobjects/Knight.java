package chessobjects;

import rules.Config;

public class Knight extends Piece {
    public Knight(int team) {
        super(team, Config.KNIGHT_POINT);
    }

    @Override
    public String toString() {
        if(super.getTeam() == 0)
            return "KNIGHT";
        return "knight";
    }
}
