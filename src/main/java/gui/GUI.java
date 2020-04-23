package gui;

import chessobjects.*;
import utils.Move;
import utils.Spot;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import javax.swing.*;
import java.io.File;
import java.lang.reflect.Array;
import java.net.URL;
import java.util.ArrayList;
import java.util.concurrent.Semaphore;
import javax.imageio.ImageIO;


class MatrixButton extends JButton {
    private int row;
    private int col;
    private boolean isActivate = false;


    public MatrixButton(int row, int col, boolean isActivate) {
        this.row = row;
        this.col = col;
        this.isActivate = isActivate;
    }

    public void setActivate(boolean activate) {
        isActivate = activate;
    }

    public boolean isActivate() {
        return isActivate;
    }

    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    @Override
    public MatrixButton clone(){
        MatrixButton matrixButton = new MatrixButton(row, col, this.isActivate);
        matrixButton.setMargin(this.getMargin());
        matrixButton.setBackground(this.getBackground());
        matrixButton.setIcon(this.getIcon());
        return matrixButton;
    }
}

public class GUI {
    Semaphore sem = new Semaphore(1);
    private ArrayList<Spot> playerClicks = new ArrayList<Spot>();
    public final JPanel gui = new JPanel();
    private MatrixButton[][] chessBoardSquares = new MatrixButton[8][8];
    private Image[][] chessPieceImages = new Image[2][6];

    public GUI() {
        try {
//            URL url = new URL("./pieces.png");
            BufferedImage bi = ImageIO.read(new File("./pieces.png"));
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 6; j++) {
                    chessPieceImages[i][j] = bi.getSubimage(j * 64, i * 64, 64, 64);
                }
            }
        } catch (Exception e) {
            System.exit(1);
        }
        JPanel chessBoard = new JPanel(new GridLayout(0, 8));
        JPanel boardConstrain = new JPanel(new GridBagLayout());
        boardConstrain.add(chessBoard);
        gui.add(boardConstrain);


        Insets buttonMargin = new Insets(0, 0, 0, 0);
        ImageIcon defaultIcon = new ImageIcon(new BufferedImage(64, 64, BufferedImage.TYPE_INT_ARGB));
        for (int row = 0; row < 8; row++){
            for (int col = 0; col < 8; col++){
                final MatrixButton matrixButton = new MatrixButton(row, col, true);
                matrixButton.addActionListener(new AbstractAction(){

                    public void actionPerformed(ActionEvent e) {
                        try{
                            sem.acquire();
                        }
                        catch (InterruptedException ex){
                            ex.printStackTrace();
                            System.exit(1);
                        }
                        if(matrixButton.isActivate()){
                            playerClicks.add(new Spot(matrixButton.getRow() + 1, matrixButton.getCol() + 1));
                            System.out.println(String.format("(%d, %d)",matrixButton.getRow() + 1, matrixButton.getCol() + 1));
                        }
                        sem.release();
                    }
                });
                matrixButton.setMargin(buttonMargin);
                matrixButton.setBackground(Color.WHITE);
                matrixButton.setIcon(defaultIcon);
                chessBoardSquares[row][col] = matrixButton;
            }
        }

        for (int row = 0; row < 8; row++)
            for (int col = 0; col < 8; col++)
                chessBoard.add(chessBoardSquares[row][col]);
    }

    private Image getImage(Piece piece){
        assert piece != null;
        int team = Math.abs(1 - piece.getTeam());
        if(piece instanceof King)
            return chessPieceImages[team][0];
        else if(piece instanceof Queen)
            return chessPieceImages[team][1];
        else if(piece instanceof Rook)
            return chessPieceImages[team][2];
        else if(piece instanceof Knight)
            return chessPieceImages[team][3];
        else if(piece instanceof Bishop)
            return chessPieceImages[team][4];
        return chessPieceImages[team][5];
    }

    public void display(Board board, int team){

        for(int row = 0; row < 8; ++row){
            for(int col = 0; col < 8; ++col){

                MatrixButton matrixButton = chessBoardSquares[row][col];

//                // set activate button for each turn
//                if(board.board[row+1][col+1] == null)
//                    matrixButton.setActivate(true);
//                else if(board.board[row+1][col+1].getTeam() == team)
//                        matrixButton.setActivate(true);
//                else
//                    matrixButton.setActivate(false);
                //set color
                if ((col % 2 == 1 && row % 2 == 1) || (col % 2 == 0 && row % 2 == 0))
                    matrixButton.setBackground(Color.WHITE);
                else
                    matrixButton.setBackground(Color.BLACK);

                // set icon
                if(board.board[row + 1][col + 1] == null)
                    matrixButton.setIcon(null);
                else
                    matrixButton.setIcon(new ImageIcon(this.getImage(board.board[row + 1][col + 1])));
            }
        }

    }

    public void displayLastMove(Spot start, Spot end){
        chessBoardSquares[start.getRow() - 1][start.getCol() - 1].setBackground(Color.YELLOW);
        chessBoardSquares[end.getRow() - 1][end.getCol() - 1].setBackground(Color.GREEN);
    }

    public void displayCheck(Spot kingPos){
        chessBoardSquares[kingPos.getRow() - 1][kingPos.getCol() - 1].setBackground(Color.RED);
    }

    public ArrayList<Spot> getUserClicks(){
        try{
            sem.acquire();
        }
        catch (InterruptedException ex){
            ex.printStackTrace();
            System.exit(1);
        }
        ArrayList <Spot> tmp = new ArrayList<Spot>();
        for(Spot spot: playerClicks)
            tmp.add(new Spot(spot.getRow(), spot.getCol()));
        sem.release();
        return tmp;
    }

    public void clearBuffer(){
        try{
            sem.acquire();
        }
        catch (InterruptedException ex){
            ex.printStackTrace();
            System.exit(1);
        }
        playerClicks.clear();
        sem.release();
    }
}
