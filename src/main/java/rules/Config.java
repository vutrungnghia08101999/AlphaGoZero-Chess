package rules;

public class Config {
    /** https://chessfox.com/free-chess-course-chessfox-com/objectives-in-chess-material-advantage/ */
    public static final int PAWN_POINT1 = 1;
    public static final int PAWN_POINT2 = 5; // in case >= 6 || <= 3
    public static final int BISHOP_POINT = 3;
    public static final int KNIGHT_POINT = 3;
    public static final int ROOK_POINT = 5;
    public static final int QUEEN_POINT = 9;
    public static final int KING_POINT = 4;
    public static final int TREE_DEPTH = 5; // using minimum-maximum algorithm
}
