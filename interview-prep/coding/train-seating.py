"""
Question: You are given a train with N seats in a row, numbered 1 to N. 
When a passenger sits on seat i, all seats in the range [i-K, i+K] become unavailable. 
Passengers randomly choose an available seat until no seats remain. 
Compute the expected number of passengers who will sit down.
"""
def expected_number_of_passengers(N, K):
    if N <= 0:
        return 0.0
    dp = {0: 0.0}
    for n in range(1, N + 1):
        term1 = (n - 1) * dp.get(n - 1, 0.0)
        term2 = 2 * dp.get(n - K - 1, 0.0)
        dp[n] = (term1 + 1 + term2) / n
    return dp[N]

print(expected_number_of_passengers(5, 1))

"""
Follow up
What if passengers aren't random but always choose a seat to maximize the remaining available seats?
"""
def max_greedy_passengers(N: int, K: int) -> int:
    """
    Calculates the maximum number of passengers that can sit using a greedy strategy.
    
    The greedy strategy is to always choose the available seat with the lowest index,
    as this maximizes the number of remaining seats for future passengers.

    Args:
        N: The total number of seats.
        K: The distance constraint.

    Returns:
        The total number of passengers who will sit.
    """
    if N <= 0:
        return 0

    # is_blocked[i] corresponds to seat i+1.
    is_blocked = [False] * N
    passenger_count = 0

    # Iterate through each seat from 1 to N.
    for i in range(N):
        # The current seat is seat number i + 1.
        
        # If the current seat is not blocked, a passenger sits here.
        if not is_blocked[i]:
            passenger_count += 1
            
            # This new passenger blocks all seats within distance K.
            # The seat index is `i`, the seat number is `i+1`.
            # The affected range is [(i+1)-K, (i+1)+K].
            # Convert back to 0-based indices for the array.
            
            start_block_idx = max(0, i - K)
            end_block_idx = min(N - 1, i + K)
            
            for j in range(start_block_idx, end_block_idx + 1):
                is_blocked[j] = True
                
    return passenger_count