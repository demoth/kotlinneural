import java.util.*
import kotlin.collections.HashMap
import kotlin.collections.LinkedHashMap

class LruCache(val capacity: Int) {

    val data: MutableMap<Int, String>
    val updated: MutableMap<Int, Long>

    init {
        data = HashMap(capacity * 100 / 75)
        updated = LinkedHashMap()
    }

    // LRU - last in the queue
    fun get(key: Int): String? {
        if (!data.containsKey(key))
            return null

        updateTime(key)

        return data[key]
    }

    fun remove(key: Int) {
        removeEntry(key)
    }

    fun put(key: Int, newValue: String) {
        if (!data.containsKey(key) && data.size == capacity) {
            val lru = updated.entries.first() // LRU??
            removeEntry(lru.key)
        }
        data[key] = newValue
        updateTime(key)
    }

    private fun updateTime(key: Int) {
        updated.remove(key)
        updated[key] = Date().time
    }

    private fun removeEntry(key: Int) {
        updated.remove(key)
        data.remove(key)
    }
}
