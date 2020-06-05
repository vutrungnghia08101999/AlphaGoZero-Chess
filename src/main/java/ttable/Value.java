package ttable;

import utils.Move;

public class Value {
	public int depth, lower, upper;
//	public Move bestMove;
	
	
	public Value(int depth) {
		super();
		this.depth = depth;
		lower = -10000;
		upper = 10000;
	}


	public Value(int depth, int lower, int upper) {
		super();
		this.depth = depth;
		this.lower = lower;
		this.upper = upper;
	}


}
