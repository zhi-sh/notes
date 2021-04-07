数组 

1. 只要看到有序数组，就考虑是否可以用二分查找；

   ~~~python
   def binary_search(arr:list, target):
     left, right = 0, len(arr)-1 # [left, right]  闭区间决定条件为 left<=right
     
    	while(left <= right):
       mid = (left+right)//2
       if target == arr[mid]:
         return mid
       elif target < arr[mid]: # target 在左区间，所以[mid - 1]
         right = mid - 1
       elif target > arr[mid]: # target 在右区间，所以[mid + 1, right]
         left = mid + 1
       
   	return -1
   
   r'''
   	时间复杂度: O(logn)
   	空间复杂度: O(1)
   '''
   ~~~

2. 在数组和链表中，双指针法是很常见的。通常可以将时间复杂度O(n^2)的解法优化为O(n)。

动态规划

1. 编辑距离

   有两个字符串X和Y，X长度为n，Y长度为m。D[i, j] 为X[1..i]到Y[1.. j]的最小编辑距离。X[1..i]表示X的前i个字符，Y[1.. j]表示Y的前j个字符，D[n,m]为X和Y的最小编辑距离。

   ![image-20210407101919437](/Users/liuzhi/Library/Application Support/typora-user-images/image-20210407101919437.png)

   ```python
   def levenshtein_distance(source, target):
       import numpy as np
       
       len_src = len(source)
       len_dst = len(target)
   
       # dp[i][j] 表示source[1..i]，target[1..j]两个子串的编辑距离
       dp = np.zeros((len_src + 1, len_dst + 1))
   
       # 初始状态
       for i in range(1, len_src + 1):
           dp[i][0] = i
       for j in range(1, len_dst + 1):
           dp[0][j] = j
   
       for i in range(1, len_src + 1):
           for j in range(1, len_dst + 1):
               if source[i - 1] == target[j - 1]:
                   dp[i][j] = dp[i - 1][j - 1]  # 相等，则不需要操作
               else:
                   dp[i][j] = min(
                       dp[i - 1][j] + 1,  # 删除
                       dp[i][j - 1] + 1,  # 插入
                       dp[i - 1][j - 1] + 1  # 替换
                   )
   
       return dp[len_src][len_dst]
   ```

   

2. 