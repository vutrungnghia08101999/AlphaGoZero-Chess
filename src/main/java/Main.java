import chessobjects.*;
import engine.*;
import gui.GUI;
import rules.Config;
import utils.Move;
import utils.Spot;

import javax.swing.*;
import java.util.ArrayList;
public class Main {
    public static void main(String[] args) throws InterruptedException {
        ArrayList<Move> history = new ArrayList<Move>();
        Board board = new Board();
        int turn = 1;
        Move move = null;
        CPU cpu = new CPU();

        GUI gui = new GUI();
        JFrame f = new JFrame("CHESS-LUSHEETA");
        f.add(gui.gui);
        f.pack();
        f.setVisible(true);

        board.displayBoard();
        gui.display(board, turn);
//        gui.display(board);
        while(true){
            ArrayList<Move> recentMoves = new ArrayList<Move>();
            int n = Math.max(0, history.size() - 9);
            for(int i = n; i < history.size(); ++i){
                Move mv = history.get(i);
                recentMoves.add(new Move(mv.getStart(), mv.getEnd(), mv.isCastling(), mv.isPromoted()));
            }

            /**************** COMPUTER VS COMPUTER *********************/
//            System.out.println("Turn: " + turn);
//            long start_time = System.currentTimeMillis();
//            move = cpu.searchNextMove(board, turn, Config.TREE_DEPTH);
//            long end_time = System.currentTimeMillis();
//            System.out.println("Runtime: " + (end_time-start_time) + "(ms)");
//            Thread.sleep(1000);

            /**************** COMPUTER VS PLAYER *********************/
            if(turn == 0){
                System.out.println("Turn: TERMINATOR - AI");
                long start_time = System.currentTimeMillis();
                move = cpu.searchNextMove(board, turn, Config.TREE_DEPTH, recentMoves);
                long end_time = System.currentTimeMillis();
                System.out.println("Runtime: " + (end_time-start_time) + "(ms)");
            }
            else{
                System.out.println("Player turn: ");
                do{
                    ArrayList<Spot> userClicks = new ArrayList<Spot>();
                    while(userClicks.size() < 2){
                        Thread.sleep(50);
                        userClicks = gui.getUserClicks();
                    }
                    for(int i = 0; i < userClicks.size() - 1; ++i){
                        Spot start = userClicks.get(i);
                        Spot end = userClicks.get(i+1);
                        Piece piece = board.board[start.getRow()][start.getCol()];
                        if(piece instanceof Pawn && end.getRow() == 1)
                            move = new Move(start, end, false, true);
                        else if(piece instanceof King && Math.abs(end.getCol() - start.getCol()) == 2)
                            move = new Move(start, end, true, false);
                        else
                            move = new Move(start, end);
                        if(cpu.isValidMove(move,board, turn))
                            break;
                        else
                            move = null;
                    }
                }while (move == null);
                gui.clearBuffer();
            }
            /**********************************************************/
            history.add(move);
            System.out.println(move);
            board = cpu.getNextState(board, move);
            System.out.println("TEAM CPU Metrics: " + cpu.evaluate(board, 0));
            board.displayBoard();
            gui.display(board, Math.abs(1- turn));
            gui.displayLastMove(move.getStart(), move.getEnd());
            if(cpu.isChecked(board, Math.abs(1-turn)))
                gui.displayCheck(Utils.getKingPosition(board, Math.abs(1-turn)));
		
            turn = Math.abs(1-turn);
            if(cpu.isCheckedMate(board, turn)){
                System.out.println(String.format("Team %d won.", Math.abs(1-turn)));
                break;
            }
            move = null;
        }
    }
}
