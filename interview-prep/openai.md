### **Problem: Train Seating with Distance Constraint**

* **Question**: You are given a train with `N` seats in a row, numbered 1 to `N`. When a passenger sits on seat `i`, all seats in the range `[i-K, i+K]` become unavailable. Passengers randomly choose an available seat until no seats remain. Compute the expected number of passengers who will sit down.
* **Solution (Dynamic Programming)**: This problem can be solved using the linearity of expectation. Let `E(n)` be the expected number of passengers for `n` available seats. We can derive a recurrence relation and solve it iteratively. The relation is `E(n) = ( (n-1)*E(n-1) + 1 + 2*E(n-K-1) ) / n` for `n > 0`, with base cases `E(j) = 0` for `j <= 0`.

    ```python
    def expected_passengers(N: int, K: int) -> float:
        """Computes the expected number of passengers using dynamic programming."""
        if N <= 0:
            return 0.0
        dp = {0: 0.0}
        for n in range(1, N + 1):
            term1 = (n - 1) * dp.get(n - 1, 0.0)
            term2 = 2 * dp.get(n - K - 1, 0.0)
            dp[n] = (term1 + 1 + term2) / n
        return dp[N]
    ```
* **Potential Follow-ups**:
    1.  **Complexity & Optimization**: Ask for time/space complexity (O(N) time, O(N) space). Ask to optimize space to O(K) using a circular array/deque.
    2.  **Large N**: If N is very large (e.g., 10^18), the O(N) solution is too slow. This hints at using **matrix exponentiation** to solve the linear recurrence in O(K³ log N) time.
    3.  **Circular Train**: How does the solution change if seat N is adjacent to seat 1? This breaks the edge effects and requires a more complex DP state, as placing a passenger always creates two subproblems.
    4.  **Greedy Passengers**: What if passengers aren't random but always choose a seat to maximize the remaining available seats? This becomes a deterministic greedy problem, not an expectation problem.
    5.  **Probability Distribution**: Instead of the expected value, find the probability of seating *exactly* `m` passengers. This requires a more complex DP state like `P(n, m)`.
    6.  **Verification**: How would you write a **Monte Carlo simulation** to verify your analytical result? This involves simulating the random process thousands of times and averaging the results.

---
---
---

## **Full List of Questions and Solutions**

Here is the complete list of questions extracted from the provided document, organized by the original document's sections.

### **High Frequency (高频)**

* **1. Bicycle and Person Matching (人车匹配)**
    * **Question**: Given `m` people and `n` bikes, match them.
        * **Version 1 (Greedy - LC1057)**: Each person gets their closest available bike.
        * **Version 2 (Optimal)**: Minimize the total distance for all assignments.
    * **Solution**:
        * **Version 1**: Use **Bucket Sort**. Calculate all `m*n` distances and group `(person, bike)` pairs by distance. Iterate through distances from smallest to largest to make assignments.
        * **Version 2**: This is the **Assignment Problem**. Model as a bipartite graph and find the minimum weight perfect matching. In an interview, use `scipy.optimize.linear_sum_assignment`.

* **2. Remove Extra Edge in Binary Tree (二叉树删除边)**
    * **Question**: A Binary Search Tree has one extra edge violating the BST property. Find and remove it.
    * **Solution**: Use **DFS with a valid range**. Pass `(min_val, max_val)` down the recursion. If `node.val` is outside the range, the edge to it is invalid; return `None` to disconnect it.

* **3. Robot from Top-Left to Top-Right (机器人左上到右上)**
    * **Question**: In an `H x W` grid, a robot moves from `(0,0)` to `(0, W-1)`. It can only move right, up-right, or down-right. Find the number of unique paths.
    * **Solution**: **Dynamic Programming with O(H) space**. Use a 1D DP array for the current column. Calculate the next column based on `new_dp[i] = dp[i-1] + dp[i] + dp[i+1]`.

* **4. Guess Word (LC843)**
    * **Question**: Interactive game to guess a secret word from a list in 10 tries, getting the number of character matches with each guess.
    * **Solution**: **Minimax Heuristic**. In each step, choose a word from the candidates that best partitions the remaining candidates, minimizing the largest possible remaining group. A simpler, often sufficient method is to pick a random candidate.

* **5. Word Pattern Match (LC890)**
    * **Question**: Find all words from a list that match a given pattern's structure (e.g., "abb" matches "mee").
    * **Solution**: **Canonical Representation**. Convert both the pattern and each word to a standard form (e.g., "abb" -> "011", "mee" -> "011"). Compare the canonical forms.

* **6. Robot Room Cleaner (扫地机器人 - LC489)**
    * **Question**: Interactive problem to clean all reachable cells in an unknown grid with a limited API.
    * **Solution**: **DFS + Backtracking**. Use a `set` of visited relative coordinates `(x, y)`. After a recursive call returns, you must backtrack the robot to its previous position and orientation.

* **7. Exam Room (考试找位子 - LC855)**
    * **Question**: Design a class to `seat()` a student farthest from any other student.
    * **Solution**: **Ordered List**. Maintain a sorted list of occupied seats. For `seat()`, check the distance to the ends (0 and N-1) and the midpoint distance between each adjacent pair of students to find the maximum gap.

* **8. Expiring HashMap (Key有过期时间的hashmap)**
    * **Question**: Implement a hash map where keys have a Time-To-Live (TTL).
    * **Solution**: **Lazy Deletion**. Use two dictionaries: one for `key -> value` and one for `key -> expiration_timestamp`. When `get(key)` is called, check if the current time has passed the expiration time. If so, delete the key from both maps.

* **9. Random Point in Rectangles (随机取点 - LC497)**
    * **Question**: Given non-overlapping rectangles, randomly pick a point from the area they cover.
    * **Solution**: **Prefix Sum + Binary Search**. Calculate a prefix sum array of the areas of the rectangles. Generate a random number up to the total area. Use binary search on the prefix sum array to find which rectangle the number falls into, then randomly pick a point within that rectangle.

* **10. Car Fleet (LC853)**
    * **Question**: Cars on a single lane cannot pass each other. Find the number of car fleets that reach the destination.
    * **Solution**: **Sort + Stack/Greedy**. Calculate the arrival time for each car. Sort cars by their starting position (closest to target first). Iterate through the cars; if a car's arrival time is later than the current fleet's arrival time, it forms a new fleet.

* **11. Hire K Workers (雇工人 - LC857)**
    * **Question**: Hire K workers to minimize total cost, where cost is based on a shared wage/quality ratio.
    * **Solution**: **Greedy + Priority Queue**. The team's cost is `(total_quality) * (max_ratio)`. Sort workers by their `wage/quality` ratio. Iterate through them, maintaining a max-heap of the `K` smallest qualities seen so far. At each step, calculate a potential cost and update the minimum.

* **12. Corner Rectangles (LC750)**
    * **Question**: Count the number of axis-aligned rectangles formed by four `1`s in a 0-1 matrix.
    * **Solution**: **Iterate over Row Pairs**. For every pair of rows `(r1, r2)`, count the number of columns `c` where `grid[r1][c]` and `grid[r2][c]` are both `1`. If the count is `k`, these `k` columns can form `k * (k - 1) / 2` rectangles with the two rows.

* **13. Bus Routes (LC815)**
    * **Question**: Find the minimum number of buses to travel from a source stop to a target stop.
    * **Solution**: **BFS on Routes**. Model each bus route as a node in a graph. An edge exists between two routes if they share a common stop. Perform a BFS starting from all routes that pass through the source stop to find the shortest path to a route that passes through the target.

* **14. Split Array into Consecutive Subsequences (LC659)**
    * **Question**: Can an array be split into one or more subsequences of consecutive integers, each of length at least 3?
    * **Solution**: **Greedy + Hash Maps**. Use two maps: `freq` (counts of numbers) and `tails` (counts of subsequences ending at a number). Greedily try to append the current number to an existing subsequence. If not possible, try to start a new one.

* **15. Throne Inheritance (王位继承)**
    * **Question**: Design a class for a monarchy with `birth`, `death`, and `getInheritanceOrder` methods.
    * **Solution**: **Tree + Pre-order Traversal (DFS)**. Model the family as a tree (dictionary of `parent -> [children]`). Keep a `set` of deceased members. The inheritance order is a simple pre-order traversal of the tree, skipping any names in the `dead` set.

* **16. Tree Isomorphism (树的同构问题 - LC951)**
    * **Question**: Determine if two binary trees are "flip equivalent" (one can be transformed into the other by flipping any number of nodes' children).
    * **Solution**: **Recursion**. Two trees are flip equivalent if their roots have the same value, AND (their left/right subtrees are equivalent) OR (their left/right subtrees are equivalent after one is flipped).

* **17. Delete Nodes And Return Forest (N叉树删node - LC1110)**
    * **Question**: Given a tree and a list of nodes to delete, return the roots of the remaining trees (a forest).
    * **Solution**: **Post-order Traversal (DFS)**. Recursively process the children first. If a node needs to be deleted, its non-null children become new roots. The function returns `None` to its parent, effectively deleting itself.

* **18. Vending Machine (可乐饮料机)**
    * **Question**: Given buttons that dispense a range of volumes `[min, max]`, can you press a sequence of buttons to guarantee the total volume is within a target range `[T_min, T_max]`?
    * **Solution**: **BFS on States**. The state is the current accumulated volume range `(current_min, current_max)`. Start BFS from `(0,0)`. If at any point a state `(c_min, c_max)` is reached where `c_min >= T_min` and `c_max <= T_max`, return true. Prune branches where `c_max > T_max`.

* **19. Generate Random Maze (生成随机迷宫)**
    * **Question**: Generate a random maze where there is a unique path between any two cells.
    * **Solution**: **Randomized DFS (Recursive Backtracker)**. Start with a grid of walls. Pick a starting cell, mark it as a path. Recursively explore its unvisited neighbors in a random order, carving a path between the current cell and the chosen neighbor.

* **20. Log Start/Finish (日志)**
    * **Question**: Design a logger with `start(id, time)` and `finish(id, time)` methods. The system should print finished logs in the order of their start times.
    * **Solution**: Use a dictionary for `in_progress` logs. When a log finishes, move it to a min-heap prioritized by its start time. Have a `print()` method that continuously pops from the heap and prints logs.

---
---

### **Miscellaneous & Dated Questions**

* **Go Game Surrounded Check**: Check if a group of stones is surrounded. **Solution**: Start DFS/BFS from empty adjacent points. If the search reaches the board edge, the group is not surrounded.
* **N-layer Graph Min Cost**: Find the minimum cost path from the first layer to the last. **Solution**: **Dijkstra's algorithm**.
* **Card Game Strategy (拿纸牌)**: Two players take cards from the left end (1, 2, or 3 cards). Find the max score for the first player. **Solution**: **Dynamic Programming**. `dp[i]` = max score one can get from the suffix `array[i:]`.
* **Center-Flip Image**: Flip a `byte[][]` image. **Solution**: First, reverse the order of the rows. Then, for each row, reverse the order of the bytes.
* **Fit String with Largest Font**: Find the max font size to fit a string on a screen. **Solution**: **Binary search** on the font size. For each `mid` size, check if the text fits.
* **Bricks Falling (打砖块 - LC803)**: Find the number of bricks that fall after each hit. **Solution**: **Reverse time**. Start with the final grid (after all hits) and add bricks back one by one. Use Union-Find to count newly connected stable bricks.
* **Meeting Rooms II (LC253)**: Find the minimum number of rooms required for a set of meetings. **Solution**: **Min-Heap**. Sort meetings by start time. Use a min-heap to track the end times of ongoing meetings.
* **Iterator of Iterators**: Interleave elements from multiple iterators. **Solution**: Use a **Queue** (`collections.deque`) to store the iterators and cycle through them.
* **Bash Brace Expansion (LC1087)**: Expand a string like "a{b,c}d" to ["abd", "acd"]. **Solution**: **Recursion/DFS**.
* **Magic Dictionary (LC676)**: Find if a word exists in a dictionary with a one-letter change. **Solution**: Pre-process by creating generic patterns for each dictionary word (e.g., "hello" -> "h*llo", "he*lo", etc.) and store them in a hash map.
* **Sentence Screen Fitting (LC418)**: How many times can a sentence be typed on a screen of given dimensions. **Solution**: **Dynamic Programming or Simulation with Memoization**. Calculate how many words fit starting from each word index.
* **Backspace String Compare (LC844)**: Compare two strings after processing backspaces. **Solution**: Use **two pointers** from the end of each string for an O(1) space solution.
* **Increasing Triplet Subsequence (LC334)**: Find if an array contains an increasing subsequence of length 3. **Solution**: Maintain two variables, `first_min` and `second_min`, in a single pass.
* **House Robber III (LC337)**: Rob houses in a binary tree without robbing adjacent ones. **Solution**: **DFS**. The recursive function should return a pair: `(max_if_robbing_this_node, max_if_not_robbing_this_node)`.
* **Convert BST to Sorted Doubly Linked List (LC426)**: **Solution**: **In-order traversal**. Keep track of the `previous` node visited and link it to the `current` node.
* **Burst Balloons (LC312)**: Find the maximum coins from bursting balloons. **Solution**: **Dynamic Programming**. `dp[i][j]` = max coins from bursting balloons in the range `(i, j)`.
* **Max Chunks To Make Sorted II (LC768)**: **Solution**: Create two arrays: `max_of_left` and `min_of_right`. A chunk can be made at index `i` if `max_of_left[i] <= min_of_right[i+1]`.
* **The Maze II (LC505)**: Find the shortest path for a ball that rolls until it hits a wall. **Solution**: **Dijkstra's algorithm** (BFS with a priority queue).
* **Sum of Distances in Tree (LC834)**: **Solution**: **Two DFS passes**. The first pass (post-order) calculates subtree sizes and distances from each node to its children. The second pass (pre-order) calculates the final distances by incorporating information from the parent.
* **Most Stones Removed (LC947)**: **Solution**: **Union-Find or DFS**. The problem is equivalent to finding the number of connected components (islands) of stones. Max removed = `total_stones - num_components`.
* **Split Array Largest Sum (LC410)**: **Solution**: **Binary search on the answer**. Search for the minimum possible value for the largest subarray sum.
* **Daily Temperatures (LC739)**: **Solution**: **Monotonic Stack**. Iterate through temperatures and use a decreasing monotonic stack to find the next warmer day.
* **My Calendar II (LC731)**: **Solution**: Store start/end points. For a new event `[s, e)`, increment a counter at `s` and decrement at `e`. Track the running count; if it ever reaches 3, there is a triple booking.
* **Google Snapshot (Data Structure Design)**: Design a data structure that supports `set(index, val)`, `get(index, version)`, and `snapshot()`. **Solution**: Use a list of dictionaries or a dictionary of `TreeMap`/`SortedDict`. For each key/index, store a map of `version -> value`. `get` performs a floor lookup on the version.
* **Compare Version Strings (LC165)**: **Solution**: Split strings by `.` and compare components numerically.
* **Confusing Number (LC1056, LC1088)**: A number that becomes a different valid number when rotated 180 degrees. **Solution**: **DFS/Backtracking** to build confusing numbers digit by digit.
* **Skip Iterator**: **Solution**: Implement an iterator wrapper. Use a hash map to store counts of numbers to be skipped. The `hasNext()` method needs to look ahead to find the next valid element.
* **Binary String Distance (Trie-based)**: Distance is length of strings after removing common prefix. Find max distance in a list. **Solution**: Insert all strings into a **Trie**. The max distance is found by exploring two different branches from a node and summing their depths from that node.
* **In-order Traversal Comparison**: Compare if two BSTs have the same in-order traversal without storing the full traversal. **Solution**: Use an **iterative in-order traversal** with two stacks, one for each tree. Compare the nodes as they are popped.
* **0-1 String Flip with Allowed List**: Find the shortest path from a start string to a target string, flipping one character at a time, using only allowed intermediate strings. **Solution**: **BFS on states**. Each string is a node in a graph.
* **Longest Consecutive 1s in Matrix**: **Solution**: **Dynamic Programming**. Use a DP table `dp[i][j][direction]` to store the length of the line ending at `(i,j)` for each of the four directions (horizontal, vertical, diagonal, anti-diagonal).


## **OpenAI Model Serving Interview Questions**

### **System Design**

1.  **Design the Real-Time Inference API for ChatGPT**
    * **Use Case**: Design the backend for `api.openai.com/v1/chat/completions` to handle millions of concurrent, streaming requests with low latency and high throughput.
    * **Key Components**: API Gateway, Continuous Batching, Kubernetes-managed GPU fleet, Distributed KV Cache Management, Server-Sent Events (SSE) for streaming.

2.  **Design a Multi-Tier Model Serving System (Pro vs. Standard)**
    * **Use Case**: Offer the same model at two service levels: a low-latency "Pro Tier" and a cost-effective "Standard Tier".
    * **Key Components**: Separate infrastructure (premium GPUs for Pro, spot instances for Standard), different configurations (small batches/FP16 for Pro, large batches/INT8 for Standard), and priority-based request routing.

3.  **Design a System for Canary Deploying New Model Versions**
    * **Use Case**: Safely roll out a new, updated model version to production by gradually shifting traffic, monitoring performance, and enabling quick rollbacks.
    * **Key Components**: A service mesh (e.g., Istio) for traffic shifting, robust side-by-side monitoring of latency and quality metrics, and automated rollback triggers.

4.  **Design a High-Throughput Offline Batch Inference System**
    * **Use Case**: A customer uploads 10 million documents to be summarized within 24 hours at the lowest possible cost.
    * **Key Components**: A workflow orchestrator (e.g., Airflow), a compute fleet using spot instances, optimization for massive batch sizes (sorting by length), and I/O from a scalable object store (e.g., S3).

5.  **Design a Rate Limiting and Overload Protection System**
    * **Use Case**: Protect the public API from abuse and traffic spikes by enforcing usage quotas and gracefully degrading service under extreme load.
    * **Key Components**: Distributed rate limiting using the Token Bucket algorithm (backed by Redis), a load shedder to drop low-priority requests, and a circuit breaker to isolate failing services.

### **Large Model Specifics**

1.  **KV Cache**: Stores attention keys/values for previous tokens to avoid recomputation, changing per-token complexity from O(n²) to O(n). Its large memory footprint is a key challenge.
2.  **Batch Size vs. Latency**: Larger batches increase throughput but also increase time-to-first-token (TTFT) latency, as requests must wait for the whole batch.
3.  **Continuous Batching**: A dynamic scheduling technique that adds new requests to a running batch as old ones finish, significantly improving GPU utilization over static batching.
4.  **Quantization**: Reducing weight precision (e.g., FP16 to INT8) to decrease memory usage and increase speed, at the cost of a potential small accuracy drop.
5.  **Training vs. Inference Workloads**: Training is throughput-oriented (large batches, backward pass), while inference is latency-sensitive (small/dynamic batches, forward pass only, KV cache is dominant).
6.  **Compute vs. Memory Bandwidth**: LLM inference is typically **memory-bandwidth bound** because the time to load the massive model weights for each token generation often exceeds the compute time.
7.  **FlashAttention**: An I/O-aware attention algorithm that avoids materializing the full attention matrix in HBM, instead using faster on-chip SRAM to reduce memory reads/writes and speed up computation for long sequences.
8.  **Fast Model Loading**: Use **model sharding**. Load model chunks in parallel from a high-throughput store (like S3) directly to the VRAM of multiple GPUs.
9.  **High TTFT, Low TPOT**: This indicates a bottleneck in the **prompt processing (prefill) stage**, which is computationally intensive, while the subsequent per-token generation is fast.
10. **Terminating a Single Request in a Batch**: A modern serving system (like one using vLLM) will remove the terminated request from the batch and free its resources (KV cache slots) without interrupting the other requests.

### **Leadership & Culture**

1.  **Dealing with Ambiguity**: Describe a time you built a complex system with evolving requirements. (Focus on modular design, iterative development, and strong communication with stakeholders).
2.  **Crisis Management**: Walk through the first 30 minutes of a production outage. (Focus on a clear process: Mitigate first (e.g., rollback), then Communicate, then Diagnose).
3.  **Cross-Functional Collaboration**: How would you evaluate a risky but high-reward proposal from a research team? (Focus on a data-driven, phased approach: PoC, offline tests, then a limited online canary test with clear success/failure metrics).
4.  **Prioritization and Impact**: Would you improve GPU utilization by 5% or build a tool that saves every engineer 2 hours/week? (Argue for the developer tool as an investment in team velocity that has compounding returns).
5.  **Motivation and Vision**: Why are you interested in the challenges of *serving* large models? What will be the biggest challenge for serving 100x larger models? (Focus on memory bandwidth, communication overhead, and the need for co-designing models and systems).