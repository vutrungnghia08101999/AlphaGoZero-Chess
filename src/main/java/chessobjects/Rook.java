package chessobjects;


import rules.Config;

public class Rook extends Piece {
    private boolean isCastlingPossible;

    public Rook(int team) {
        super(team, Config.ROOK_POINT);
        this.isCastlingPossible = true;
    }

    public void setCastlingPossible(boolean castlingPossible) {
        isCastlingPossible = castlingPossible;
    }

    public boolean isCastlingPossible() {
        return isCastlingPossible;
    }

    @Override
    public String toString() {
        if(super.getTeam() == 0)
            return "ROOK";
        return "rook";
    }
}
