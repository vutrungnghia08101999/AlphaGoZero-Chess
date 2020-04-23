package chessobjects;

import rules.Config;

public class King extends Piece {
    private boolean isCastlingPossible;
    private boolean isCastlingDone;
    public King(int team) {
        super(team, Config.KING_POINT);
        this.isCastlingPossible = true;
        this.isCastlingDone = false;
    }

    public boolean isCastlingPossible() {
        return isCastlingPossible;
    }

    public boolean isCastlingDone() {
        return isCastlingDone;
    }

    public void setCastlingDone(boolean castlingDone) {
        isCastlingDone = castlingDone;
    }

    public void setCastlingPossible(boolean castlingPossible) {
        isCastlingPossible = castlingPossible;
    }

    @Override
    public String toString() {
        if(super.getTeam() == 0)
            return "KING";
        return "king";
    }
}
