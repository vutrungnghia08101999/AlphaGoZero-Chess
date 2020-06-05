import chessobjects.Board;
import chessobjects.Pawn;
import chessobjects.Piece;
import engine.CPU;
import rules.Config;
import utils.Move;
import utils.Spot;


import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class UCIInterface{
    public static void main(String[] args) {
        UCI uci = new UCI();
        uci.uciCommunication();
    }
}

class UCI {
    private Board board = null;
    private int turn;
    CPU cpu = new CPU();
    UCIMove uciMove = new UCIMove();
    private String ENGINE_NAME = "Lusheeta-Engine";
    private String AUTHOR = "Lusheeta";

    public void uciCommunication() {
        Scanner input = new Scanner(System.in);
        while(true) {
            String inputString = input.nextLine();
            if ("uci".equals(inputString)) inputUCI();
            else if ("isready".equals(inputString)) isReady();
            else if ("ucinewgame".equals(inputString)) newGame();
            else if (inputString.startsWith("position"))newPosition(inputString);
            else if (inputString.startsWith("go")) go();
            else if ("quit".equals(inputString)) {
                quit();
                break;
            }
        }
        input.close();
    }

    private void inputUCI() {
        System.out.println("id name " + this.ENGINE_NAME);
        System.out.println("id author " + this.AUTHOR);
        // options if any
        this.board = new Board();
        this.turn = 1;
        System.out.println("uciok");
//        this.board.displayBoard();
    }

    private void isReady() {
        this.board = new Board();
        this.turn = 1;
        System.out.println("readyok");
//        this.board.displayBoard();
    }

    private void newGame() {
        this.board = new Board();
//        this.board.displayBoard();
    }

    private void go() {
        Move mv = cpu.searchNextMove(this.board, this.turn, Config.TREE_DEPTH);
        this.board = this.cpu.getNextState(this.board, mv);
        this.turn = 1 - this.turn;
//        System.out.println(mv);
        String move = this.uciMove.indexToUci(mv);
        System.out.println("bestmove "+ move);
//        this.board.displayBoard();
    }

    private void quit() {
        System.out.println("Good game");
//        this.board.displayBoard();
    }

    private void newPosition(String input) {
        input=input.substring(9).concat(" ");
        if (input.contains("startpos ")) {
            input = input.substring(9);
            this.board = new Board();
            this.turn = 1;
        }

        if (input.contains("moves")) {
            input = input.substring(input.indexOf("moves") + 6);
            String[] moves = input.split("\\s+");
            for (int i = 0; i < moves.length; ++i){
                String mv = moves[i];
                Move move = this.uciMove.uciToIndex(mv);
                Spot start = move.getStart();
                Spot end = move.getEnd();
                Piece piece = this.board.board[start.getRow()][start.getCol()];
                if(piece instanceof Pawn && (end.getRow() == 1 || end.getRow() == 8))
                    move = new Move(start, end, false, true);
                if (cpu.isValidMove(move, this.board, this.turn)){
                    this.board = cpu.getNextState(this.board, move);
                    this.turn = 1 - this.turn;
                }
                else{
                    this.board = new Board();
                    this.turn = 1;
                    continue;
                }
            }
        }

//        this.board.displayBoard();
    }
}

class UCIMove{
    private Map<Character, Integer> uciToIndexRow = new HashMap<Character, Integer>();
    private Map<Character, Integer> uciToIndexCol = new HashMap<Character, Integer>();
    private Map<Integer, Character> indexToUciRow = new HashMap<Integer, Character>();
    private Map<Integer, Character> indexToUciCol = new HashMap<Integer, Character>();

    public UCIMove(){
        uciToIndexRow.put('1', 8);
        uciToIndexRow.put('2', 7);
        uciToIndexRow.put('3', 6);
        uciToIndexRow.put('4', 5);
        uciToIndexRow.put('5', 4);
        uciToIndexRow.put('6', 3);
        uciToIndexRow.put('7', 2);
        uciToIndexRow.put('8', 1);

        uciToIndexCol.put('a', 1);
        uciToIndexCol.put('b', 2);
        uciToIndexCol.put('c', 3);
        uciToIndexCol.put('d', 4);
        uciToIndexCol.put('e', 5);
        uciToIndexCol.put('f', 6);
        uciToIndexCol.put('g', 7);
        uciToIndexCol.put('h', 8);

        indexToUciRow.put(1, '8');
        indexToUciRow.put(2, '7');
        indexToUciRow.put(3, '6');
        indexToUciRow.put(4, '5');
        indexToUciRow.put(5, '4');
        indexToUciRow.put(6, '3');
        indexToUciRow.put(7, '2');
        indexToUciRow.put(8, '1');

        indexToUciCol.put(1, 'a');
        indexToUciCol.put(2, 'b');
        indexToUciCol.put(3, 'c');
        indexToUciCol.put(4, 'd');
        indexToUciCol.put(5, 'e');
        indexToUciCol.put(6, 'f');
        indexToUciCol.put(7, 'g');
        indexToUciCol.put(8, 'h');

    }
    public Move uciToIndex(String move){
        char[] s = move.toCharArray();
        int startRow = uciToIndexRow.get(s[1]);
        int startCol = uciToIndexCol.get(s[0]);
        int endRow = uciToIndexRow.get(s[3]);
        int endCol = uciToIndexCol.get(s[2]);
        return new Move(new Spot(startRow, startCol), new Spot(endRow, endCol));
    }

    public String indexToUci(Move move){
        Spot start = move.getStart();
        Spot end = move.getEnd();
        char startRow = indexToUciRow.get(start.getRow());
        char startCol = indexToUciCol.get(start.getCol());
        char endRow = indexToUciRow.get(end.getRow());
        char endCol = indexToUciCol.get(end.getCol());

        String s = Character.toString(startCol) + Character.toString(startRow) + Character.toString(endCol) + Character.toString(endRow);
        if (move.isPromoted())
            s = s + "q";
        return s;
    }
}