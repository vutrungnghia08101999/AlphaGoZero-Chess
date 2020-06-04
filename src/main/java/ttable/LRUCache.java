package ttable;
import java.util.HashMap;

public class LRUCache<K, V> {
	public NodeTT<K, V> head, tail;
	public HashMap<K, NodeTT<K, V>> map = null;
	int cap = 0;
	
	public LRUCache(int capacity) {
		this.cap = capacity;
	}
		
	public V get(K key) {
		if (map.get(key) == null) {
			return null;
		}
		
		//move to tail
		NodeTT<K, V> t = map.get(key);
		
		removeNode(t);
		offerNode(t);
		
		return t.value;
	}
	
	public void put(K key, V value) {
		if (map.containsKey(key)) {
			NodeTT<K, V> t = map.get(key);
			t.value = value;
			
			removeNode(t);
			offerNode(t);
		} else {
			if (map.size() >= cap) {
				// delete head
				map.remove(head.key);
				removeNode(head);
			}
			NodeTT<K, V> node = new NodeTT<K, V>(key, value);
			offerNode(node);
			map.put(key, node);
		}
	}
	private void removeNode(NodeTT<K, V> n) {
		if (n.prev != null) {
			n.prev.next = n.next;
		} else {
			this.head = n.next;
		}
		if (n.next != null) {
			n.next.prev = n.prev;
		} else {
			this.tail = n.prev;
		}
	}
	private void offerNode(NodeTT<K, V> n) {
		if (tail != null) {
			tail.next = n;
		}
		n.prev = tail;
		n.next = null;
		tail = n;
		if (head == null) {
			head = tail;
		}
	}
}