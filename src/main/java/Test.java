import java.util.HashMap;

import chessobjects.Board;
import ttable.LRUCache;
import ttable.NodeTT;
import ttable.Value;

public class Test {
	public int a, b;
	public Test(int a, int b) {
		this.a = a;
		this.b = b;
	}
	@Override
	public boolean equals(Object tt) {
		Test t = (Test) tt;
		if (t.a == this.a && t.b == this.b)
			return true;
		return false;
	}
//	@Override
//	public int hashCode() {
//		return a + b;
//	}
	public static long pow2(long a, long b) {
        long re = 1;
        while (b > 0) {
            if ((b & 1) == 1) {
                re *= a;        
            }
            b >>= 1;
            a *= a; 
        }
        return re;
    }
	public static void main(String[] args) {
		Board b1 = new Board();
		Board b2 = b1.clone();
		Board b3 = new Board();
		
		LRUCache<Board, Value> san = new LRUCache<Board, Value>(10000);
		san.map = new HashMap<Board, NodeTT<Board, Value>>();
		san.put(b1, new Value(3));
		if (san.get(b3) != null)
			System.out.println(san.get(b3));
		
	}
	
}