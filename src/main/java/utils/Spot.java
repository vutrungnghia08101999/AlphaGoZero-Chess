package utils;


public class Spot {
    private int row;
    private int col;

    public Spot(int row, int col) {
        this.row = row;
        this.col = col;
    }


    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    @Override
    public boolean equals(Object object) {
        Spot spot = (Spot)object;

        return spot.getCol() == this.getCol() && spot.getRow() == this.getRow();
    }

    @Override
    public String toString() {
        return "(" + row + ", " + col + ")";
    }
}
