import chessobjects.Board;
import chessobjects.King;
import chessobjects.Pawn;
import chessobjects.Piece;
import engine.CPU;
import rules.Config;
import utils.Move;
import utils.Spot;

import java.util.ArrayList;
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
    private ArrayList<Move> history = new ArrayList<Move>();
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
        System.out.println("uciok");
    }

    private void isReady() {
        System.out.println("readyok");
    }

    private void newGame() {
        this.board = new Board();
//        this.board.displayBoard();
    }

    private void go() {
        ArrayList<Move> recentMoves = new ArrayList<Move>();
        int n = Math.max(0, this.history.size() - 9);
        for(int i = n; i < this.history.size(); ++i){
            Move mv = this.history.get(i);
            recentMoves.add(new Move(mv.getStart(), mv.getEnd(), mv.isCastling(), mv.isPromoted()));
        }

        Move mv = cpu.searchNextMove(this.board, this.turn, Config.TREE_DEPTH, recentMoves);
        String move = this.uciMove.indexToUci(mv);
        System.out.println("bestmove "+ move);
    }

    private void quit() {
        System.out.println("Good game");
    }

    private void newPosition(String input) {
        this.history.clear();
        input = input.substring(9).concat(" ");
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
                else if(piece instanceof King && end.getCol() - start.getCol() == 2)
                    move = new Move(start, end, true, false);

                if (cpu.isValidMove(move, this.board, this.turn)){
                    this.history.add(move);
                    this.board = cpu.getNextState(this.board, move);
                    this.turn = 1 - this.turn;
                }
                else{
                    System.out.println("Invalid move: " + mv + " - " + move);
                    this.board = new Board();
                    this.turn = 1;
                    break;
                }
            }
        }
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