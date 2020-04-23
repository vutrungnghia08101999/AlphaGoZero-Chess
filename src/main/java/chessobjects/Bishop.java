package chessobjects;


import rules.Config;

public class Bishop extends Piece {
    public Bishop(int team) {
        super(team, Config.BISHOP_POINT);
    }

    @Override
    public String toString() {
        if(super.getTeam() == 0)
            return "BISHOP";
        return "bishop";
    }
}
