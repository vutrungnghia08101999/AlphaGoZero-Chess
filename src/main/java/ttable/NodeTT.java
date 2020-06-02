package ttable;

public class NodeTT<K,V> {
	public K key;
	public V value;
	
	public NodeTT<K, V> next, prev;
	
	public NodeTT(K key, V value) {
		this.key = key;
		this.value = value;
	}
}
