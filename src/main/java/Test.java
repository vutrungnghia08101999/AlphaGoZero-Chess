import java.util.HashMap;

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
		Test san = new Test(1, 2);
		if (san.equals(new Test(1, 2)))
			System.out.println("San an com");
		HashMap<Test, Integer> map = new HashMap<Test, Integer>();
		map.put(san, 12);
		
		if (map.containsKey(new Test(1, 2))) 
			System.out.println("OK - 1");
		if (map.containsKey(san)) 
			System.out.println("Ok - 2");
		System.out.println(Test.pow2(2, 32));
	}
	
}