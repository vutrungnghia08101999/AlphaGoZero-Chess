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
		
	}
	
}