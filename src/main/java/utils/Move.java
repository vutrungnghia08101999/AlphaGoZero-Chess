package utils;

public class Move {
    private Spot start;
    private Spot end;
    private boolean isCastling;
    private boolean isPromoted;

    public Move(Spot start, Spot end) {
        this.start = start;
        this.end = end;
        this.isCastling = false;
        this.isPromoted = false;
    }

    public Move(Spot start, Spot end, boolean isCastling, boolean isPromoted) {
        this.start = start;
        this.end = end;
        this.isCastling = isCastling;
        this.isPromoted = isPromoted;
    }

    public boolean isCastling() {
        return isCastling;
    }

    public boolean isPromoted() {
        return isPromoted;
    }

    public Spot getStart() {
        return start;
    }

    public Spot getEnd() {
        return end;
    }

    @Override
    public String toString() {
        return String.format("(%d, %d) => (%d, %d) - Castling: %s - Promoted: %s",
                start.getRow(),
                start.getCol(),
                end.getRow(),
                end.getCol(),
                this.isCastling(),
                this.isPromoted());
    }

    @Override
    public boolean equals(Object obj) {
        assert obj != null;
        Move move = (Move) obj;
        return this.start.equals(move.getStart()) && this.end.equals(move.getEnd()) && this.isCastling() == move.isCastling() && this.isPromoted() == move.isPromoted();
    }
}
