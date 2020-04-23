package chessobjects;

public abstract class Piece {
    private int team;
    private int point;

    public Piece(int team, int point) {
        this.team = team;
        this.point = point;
    }

    public int getTeam() {
        return team;
    }

    public int getPoint() {
        return point;
    }
}
