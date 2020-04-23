package chessobjects;


import rules.Config;

public class Pawn extends Piece {
    public Pawn(int team) {
        super(team, Config.PAWN_POINT1);
    }


    public int getPoint(int row) {
        int team = super.getTeam();
        if(team == 0){
            if(row >= 6)
                return Config.PAWN_POINT2;
        }
        else{
            if(row <= 3)
                return Config.PAWN_POINT2;
        }
        return Config.PAWN_POINT1;
    }

    @Override
    public String toString() {
        if(super.getTeam() == 0)
            return "PAWN";
        return "pawn";
    }
}
