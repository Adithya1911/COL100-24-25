
# Lab 8

# problem 1
def num_allocations(H):
    if len(H) == 0:
        return 1
    ans = 0
    for i in range(1, len(H)):
        if H[i] != H[0]:
            ans += num_allocations(H[1:i] + H[(i+1):])
    return ans
    
def solution(inp):
    return num_allocations(inp)

# problem 2

def helper(x, num):
    if num == 0:
        return [[]]
    if x > num:
        return []
    ret = []
    # for i in range(x, num + 1):
    for i in range(num, x - 1, -1):
        R = helper(i, num - i)
        ret += [[i]+x for x in R]
    return ret
def partitions(N):
    return helper(1, N)



# Lab 9

# problem1

def minimum_amount(L, X):
    if X <= 0:
        return 0
    ret = L[0] * X
    for v in L:
        ret = min(ret, v + minimum_amount(L, X - v))
    return ret

def solution(inp):
    return minimum_amount(*inp)


# Lab 10
# problem 1
import functools
def num_ways(R, G, B):
    # r red balls, g green balls, b blue balls, such that the first ball is red
    @functools.lru_cache(None)
    def constrained_num_ways(r, g, b):
        if r == 0: return 0
        if r + g + b == 1: return 1
        return constrained_num_ways(g, r - 1, b) + constrained_num_ways(b, r - 1, g)
    return constrained_num_ways(R, G, B) + constrained_num_ways(G, B, R) + constrained_num_ways(B, R, G)

def solution(inp):
    return num_ways(*inp)


# problem 2
import functools
def maximum_reward(A):
    n, m = len(A), len(A[0])
    @functools.lru_cache(None)
    def recurse(i, j):
        answer = A[i][j]
        if i < n - 1:
            answer = max(answer, A[i][j] + recurse(i + 1, j))
        if j < m - 1:
            answer = max(answer, A[i][j] + recurse(i, j + 1))
        return answer
    return recurse(0, 0)

def solution(inp):
    return maximum_reward(inp)

# problem 3
def distinct_frequencies(L):
    D = {}
    for x in L:
        if x in D:
            D[x] += 1
        else:
            D[x] = 1
    F = {}
    for u in D:
        F[D[u]] = 1
    return [f for f in F]

def solution(inp):
    return sorted(distinct_frequencies(inp))


# Lab 11

# problem 1   
def power_factorial(n, p):
    try:
        if p <= 0:
            raise ValueError("INVALID VALUE OF P")
        if not is_prime(p):
            raise ValueError("INVALID VALUE OF P")

        n //= p
        ret = 0
        
        while n > 0:
            ret += n
            n //= p
        return ret
    
    except Exception as e:
        return f"{e}"


# problem 2
def safe_division(array, first_index, second_index):

    # Check if the indices are out of bounds
    try:
        first_number = array[first_index]
        second_number = array[second_index]
    except IndexError:
        return "INDEX OUT OF BOUND"

    # Try to convert the elements at the indices to integers
    try:
        num1 = int(first_number)
        num2 = int(second_number)
    except ValueError:
        return "INVALID NUMBER"

    # Check for division by zero
    try:
        result = round(num1 / num2, 2)
    except ZeroDivisionError:
        return "DIVISION BY ZERO"
    
    return result


# problem 3
def minimum_training_difficulty(A, m):
    saved_answers = {}
    def helper(last, days):
        if days == 0:
            if last == 0:
                return 0
            else:
                return -1
        if (last, days) in saved_answers:
            return saved_answers[(last, days)]
        answer = -1
        for prev in range(0, last):
            prev_difficulty = helper(prev, days - 1)
            added_difficulty = (A[last] - A[prev])**2
            if prev_difficulty != -1:
                total_difficulty = prev_difficulty + added_difficulty
                answer = total_difficulty if answer == -1 else min(answer, total_difficulty)
        saved_answers[(last, days)] = answer
        return answer
    return helper(len(A) - 1, m)

def solution(inp):
    return minimum_training_difficulty(*inp)



# Lab 12

# problem 1
def maximum_total_value(V):
    memo = {}
    def helper(ind):
        if ind < 0:
            return 0
        if ind in memo:
            return memo[ind]
        memo[ind] = max(helper(ind - 1), V[ind] + helper(ind - 2))
        return memo[ind]
    return helper(len(V) - 1)

def solution(inp):
    return maximum_total_value(inp)


# problem 2
def maximum_total_value(V):
    memo = {}
    def helper(ind):
        if ind < 0:
            return 0
        if ind in memo:
            return memo[ind]
        memo[ind] = max(helper(ind - 1), V[ind] + helper(ind - 2))
        return memo[ind]
    return helper(len(V) - 1)

def maximum_total_value_circle(V):
    if len(V) == 1: return V[0]
    if len(V) == 2: return max(V)
    return max(V[0] + maximum_total_value(V[2:-1]), maximum_total_value(V[1:]))


# problem 3
def minimum_training_difficulty(A, m):
    saved_answers = {}
    def helper(last, days):
        if days == 0:
            if last == 0:
                return 0
            else:
                return -1
        if (last, days) in saved_answers:
            return saved_answers[(last, days)]
        answer = -1
        for prev in range(0, last):
            prev_difficulty = helper(prev, days - 1)
            added_difficulty = (A[last] - A[prev])**2
            if prev_difficulty != -1:
                total_difficulty = prev_difficulty + added_difficulty
                answer = total_difficulty if answer == -1 else min(answer, total_difficulty)
        saved_answers[(last, days)] = answer
        return answer
    return helper(len(A) - 1, m)

def solution(inp):
    return minimum_training_difficulty(*inp)


# Lab 13

# problem 1

def count_zero_sum_intervals(L):
    memo = {}
    memo[0] = 1
    sm = 0
    ans = 0
    for i in range(len(L)):
        sm += L[i]
        if sm in memo:
            ans += memo[sm]
            memo[sm] += 1
        else:
            memo[sm] = 1
    return ans


# problem 2
def count_zero_sum_intervals(L):
    memo = {}
    memo[0] = 1
    sm = 0
    ans = 0
    for i in range(len(L)):
        sm += L[i]
        if sm in memo:
            ans += memo[sm]
            memo[sm] += 1
        else:
            memo[sm] = 1
    return ans

def count_balanced_sub_matrices(matrix):
    n = len(matrix)
    answer = 0
    for i in range(n):
        sums = [0] * n
        for j in range(i, n):
            for k in range(n):
                sums[k] += -1 if matrix[j][k] == 'B' else 1
            answer += count_zero_sum_intervals(sums)
    return answer

def solution(inp):
    return count_balanced_sub_matrices(inp)



# Lab 14
# problem 1
def num_dist_leq_k(x, d):
    j, n = 0, len(x)
    num = 0
    for i in range(n):
        while j < i and x[i] - x[j] > d:
            j += 1
        num += i - j
    return num

def solution(inp):
    return num_dist_leq_k(*inp)


# problem 2
def get_seat_numbers(S, n):
    L = len(S)
    ret = []
    for i in range(n):
        old, max_privacy, pos = 0, -1, -1
        for j in range(1, L):
            if S[j] == '#':
                diff = j - old
                privacy = -1 if diff == 1 else (diff - 2) // 2
                if privacy > max_privacy:
                    max_privacy = privacy
                    pos = (old + j) // 2
                old = j
        S = S[:pos] + '#' + S[pos+1:]
        ret += [pos]
    return ret


