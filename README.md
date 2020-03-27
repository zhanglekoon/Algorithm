# 剑指offer
## 数组部分
### 03 数组中重复的数字
找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
```
class Solution {
    public int findRepeatNumber(int[] nums) {
         int len = nums.length;
         int[] arr = new int[len];
         for(int i=0;i<len;i++)
         {
             arr[i] = 0;
         }
        for(int i=0;i<len;i++)
         {
            arr[nums[i]]++;
            if(arr[nums[i]]>1)
            return nums[i];
         }
         return 0;
    }
}
```
### 04 二维数组中的查找
在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        //可以利用左下角或者右上角的元素作为flag，利用消元思想提高效率，避免使用暴力法
        int row = matrix.length; 
        if(row==0) return false;
        int col = matrix[0].length;
        if(col==0) return false;
        int flag = matrix[0][col-1]; //利用右上角元素作为flag
        int i = 0;
        int j = col-1;
        while(i>=0&&j>=0&&i<row&&j<col)
        {
            if(target>matrix[i][j]) i++;
           else if(target<matrix[i][j]) j--;
           else
            return true;
        }
        return false;

    }
}
```
### 面试题11. 旋转数组的最小数字
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

```
class Solution {
    public int minArray(int[] numbers) {
    if(numbers.length==0) return 0;
    if(numbers.length==1) return numbers[0];
    int res = Integer.MIN_VALUE;
    for(int i=1;i<numbers.length;i++)
    {
        if(numbers[i]<numbers[i-1])
        res = numbers[i];
    }
    return res==Integer.MIN_VALUE?numbers[0]:res;
    }
}
```
### 29 顺时针打印矩阵
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
```
##剑指offer
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> printMatrix(int [][] array) {
        ArrayList<Integer> result = new ArrayList<Integer> ();
        if(array.length==0) return result;
        int n = array.length,m = array[0].length;
        if(m==0) return result;
        int layers = (Math.min(n,m)-1)/2+1;//这个是层数
        for(int i=0;i<layers;i++){
            for(int k = i;k<m-i;k++) result.add(array[i][k]);//左至右
            for(int j=i+1;j<n-i;j++) result.add(array[j][m-i-1]);//右上至右下
            for(int k=m-i-2;(k>=i)&&(n-i-1!=i);k--) result.add(array[n-i-1][k]);//右至左
            for(int j=n-i-2;(j>i)&&(m-i-1!=i);j--) result.add(array[j][i]);//左下至左上
        }
        return result;      
    }
}
## leetcode
class Solution {
    public int[] spiralOrder(int[][] matrix) {
         if(matrix.length==0)
         return new int[0];
         if(matrix[0].length==0)
         return new int[0];
         ArrayList<Integer> array = new ArrayList<>();
         int i=0,j=0,m=matrix.length,n=matrix[0].length;
         while(i<=(m-1)/2&&j<=(n-1)/2)
         {
             //从左到右
             for(int col=j;col<=n-1-j;col++)
             array.add(matrix[i][col]);
             //从上到下
             for(int row=i+1;row<=m-i-1;row++)
             array.add(matrix[row][n-1-j]);
             //从右到左
             for(int col=n-2-j;col>=j&&(m-i-1!=i);col--) //避免行重复
             array.add(matrix[m-i-1][col]);
             //从下到上
             for(int row=m-2-i;row>=i+1&&(n-1-j!=j);row--) //避免列重复
             array.add(matrix[row][j]);
            i++;
            j++;
         }
         int len = array.size();
         int[] res = new int[len];
         for(int u=0;u<len;u++)
         res[u] = array.get(u);
        return res; 
    }
}
```
### 面试题21. 调整数组顺序使奇数位于偶数前面
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
```
class Solution {
    public int[] exchange(int[] nums) {
        //利用双指针 left表示偶数（从头开始） right表示奇数（从后开始）
        int left = 0;
        int right = nums.length-1;
        while(left<right)
        {
            if((nums[left]&1)!=0)// 一个数&1==0 代表其为偶数
            {
                left++;
                continue;
            }
            if((nums[right]&1)!=1)
            {
                right--;
                continue;
            }
            else
            {
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
            }
        } 
        return nums;    }
}
```
### 面试题39. 数组中出现次数超过一半的数字
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
```
class Solution {
    public int majorityElement(int[] nums) {
        if(nums.length==1) return nums[0];
        HashMap<Integer,Integer> map = new HashMap<>();
        int res = 0;
        for(int i:nums)
        {
            if(map.containsKey(i))
           {
            map.put(i,map.get(i)+1);
            if(map.get(i)>nums.length/2)
           {
            res = i;
            break;
           } 
           } 
            else
            map.put(i,1);
        }
        return res;
    }
}
```
### 53 在排序数组中查找数字 I
统计一个数字在排序数组中出现的次数。
```
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length == 0) return 0;
        int flag = binarysearch(nums,target);
        if(flag==-1) return 0;
        else{
            int i=flag-1,j=flag+1,count=1;
            while(i>=0&&nums[i]==target)
            {
                count++;
                i--;
            }
            while(j<nums.length&&nums[j]==target)
            {
                count++;
                j++;
            }
            return count;
        }
    }
    public int binarysearch(int[] nums, int target){
        if(nums.length==0) return -1;
        int left = 0,right = nums.length-1;
        int mid = 0;
        while(left<=right)
        {
            mid = left+(right-left)/2;
            if(nums[mid]>target) right = mid-1;
            else if(nums[mid]<target) left = mid+1;
            else return mid;
        }
        return -1;
    }
}
```
### 53  II. 0～n-1中缺失的数字
一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

```
class Solution {
    public int missingNumber(int[] nums) {
        int len = nums.length;
        int[] result = new int[len+1];
        for(int i=0;i<len;i++){
            result[nums[i]]=1;
        }
        int res = 0;
        for(int i=0;i<len+1;i++)
        {
            if(result[i]!=1)
            res = i;
        }
        return res;
    }
}
```
### 面试题64. 求1+2+…+n
求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
```
class Solution {
    public int sumNums(int n) {
        int sum = n;
	//注意逻辑运算符的短路特性
        boolean b = (n > 0) && ((sum += sumNums(n - 1)) > 0);
        return sum;
    }
}
```
### 面试题62. 圆圈中最后剩下的数字
0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
```
//寻找删除规律
class Solution {
    public int lastRemaining(int n, int m) {
        if(n==0||m==0) return -1;
        ArrayList<Integer> res = new ArrayList<>();
        int clear = (m-1)%n;
        for(int i=0;i<n;i++)
        {
            res.add(i);
        }
        while(res.size()>1)
        {
            res.remove(clear);
            clear = (clear+m-1)%res.size();
        }
        return res.get(0);
    }
}
```
### 面试题58 - I. 翻转单词顺序
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

```
class Solution {
    public String reverseWords(String s) {
        String[] str = s.split(" ");
        StringBuilder res = new StringBuilder();
        for(int i=str.length-1;i>=0;i--)
        {
            String temp = str[i].trim();
            if (!temp.equals(""))//注意split拆分之后有一些空字符串 可能会导致结果有多余的空格 所以此处作此判断 
            {
            res.append(temp);
            res.append(' ');
            }
        }
        return res.toString().trim();
    }
}
```
### 面试题65. 不用加减乘除做加法
```
不用加减乘除做加法的方法是使用按位异或和按位与运算。计算a + b等价于计算(a ^ b) + ((a & b) << 1)，其中((a & b) << 1)表示进位。因此令a等于(a & b) << 1，令b等于a ^ b，直到a变成0，然后返回b。
class Solution {
    public int add(int a, int b) {
        while(a!=0){
            int temp = a^b;
            a =(a&b)<<1;
            b = temp;
        }
        return b;
    }
}
```
### 面试题61. 扑克牌中的顺子
从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
```
class Solution {
    public boolean isStraight(int[] nums) {
        //对数组排序，然后关注0的个数，递增的数据若有空缺，则需要nums[i]-nums[i-1]-1个0来填充
        if(nums.length!=5) return false;
       ArrayList<Integer> arr = new ArrayList<>();
       for(int i:nums)
       {
           arr.add(i);
       }
        Collections.sort(arr);
        for(int i=0;i<5;i++)
        {
            nums[i] = arr.get(i);
                    }
        int zero = 0;
        for(int i=0;i<5;i++)
        {
            if(nums[i]==0)
            {
                zero++;
                continue;
            }
           else if((i>0)&&(nums[i]==nums[i-1]))
           {
            return false;
           } 
            else if((i>0)&&(nums[i-1]!=0))//注意此处的坑 nums[i-1]如果为0,则无需进行下面
            {
                zero -= nums[i]-nums[i-1]-1;
            }
        }
        return zero>=0;
    }
}
```

## 栈
### 09 用两个栈实现队列
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
```
class CQueue {
    Stack<Integer> stack1;
    Stack<Integer> stack2;
    //考虑一个栈作为输入，另一个栈倒序输入第一个栈的内容完成队列功能
    public CQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }
    
    public void appendTail(int value) {
        stack1.push(value);
    }
    
    public int deleteHead() {
        //如果stack2中有值 则出栈
        if(!stack2.isEmpty()) return stack2.pop();
        //如果stack2没有，则把stack1的都拿走
        while(!stack1.isEmpty())
        {
            int temp = stack1.pop();
            stack2.push(temp);
        }
        if(!stack2.isEmpty())
        return stack2.pop();
        else
        return -1;
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */
```
### 30 包含min的最小栈
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
```
class MinStack {
    //因为题目要求min的复杂度为o(1),则考虑利用辅助栈 辅助栈需要与存储栈数量相同，但栈顶要保持最小
    /** initialize your data structure here. */
    Stack<Integer> stack1;
    Stack<Integer> stack2;
    public MinStack() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }
    
    public void push(int x) {
        stack1.push(x);
        if(stack2.isEmpty())
        stack2.push(x);
        else
        {
            int temp = stack2.peek()>x?x:stack2.peek();
            stack2.push(temp);
        }
    }
    
    public void pop() {
        stack1.pop();
        stack2.pop();
    }
    
    public int top() {
        return stack1.peek();
    }
    
    public int min() {
        return stack2.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```
### 面试题31. 栈的压入、弹出序列
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

思路很简单，我们尝试按照 popped 中的顺序模拟一下出栈操作，如果符合则返回 true，否则返回 false。这里用到的贪心法则是如果栈顶元素等于 popped 序列中下一个要 pop 的值，则应立刻将该值 pop 出来。

我们使用一个栈 st 来模拟该操作。将 pushed 数组中的每个数依次入栈，同时判断这个数是不是 popped 数组中下一个要 pop 的值，如果是就把它 pop 出来。最后检查栈是否为空。

```
贪心算法  
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int len = popped.length;
        int j = 0;
        for(int i=0;i<len;i++)
        {
            stack.push(pushed[i]);
            while(!stack.isEmpty()&&j<len&& (popped[j]==stack.peek()))
            {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }
}
```
### 59 滑动窗口最大值
给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
```
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if(nums.length<k||nums.length==0) return new int[0];
        for(int i=k-1;i<nums.length;i++)
        {
            res.add(getmax(nums,i,k));
        }
        int len = res.size();
        int[] result = new int[len];
        for(int j=0;j<len;j++)
        {
            result[j] = res.get(j);
        }
        return result;
    }
    public int getmax(int[] nums,int i,int k)
    {
        int temp = Integer.MIN_VALUE;
        for(int j=i-k+1;j<=i;j++)
        {
            if(nums[j]>temp)
            temp = nums[j];
        }
        return temp;
    }
}
```
### 59 II. 队列的最大值
请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

```
//注意双向队列的使用 
class MaxQueue {
    Queue<Integer> queue;
    Deque<Integer> maxqueue;
    public MaxQueue() {
        queue = new ArrayDeque<>();
        maxqueue = new ArrayDeque<>();
    }
    
    public int max_value() {
        if(maxqueue.isEmpty()) return -1;
        return maxqueue.peek();
    }
    
    public void push_back(int value) {
        queue.add(value);
        while(!maxqueue.isEmpty()&&maxqueue.getLast()<value)
        {
            maxqueue.pollLast();
        }
        maxqueue.add(value);
    }
    
    public int pop_front() {
        if(queue.isEmpty()) return -1;
        int res = queue.poll();
        if(maxqueue.peek()==res)
       { maxqueue.poll();}
        return res;
    }
}

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue obj = new MaxQueue();
 * int param_1 = obj.max_value();
 * obj.push_back(value);
 * int param_3 = obj.pop_front();
 */
```
### 面试题66. 构建乘积数组
给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
```
class Solution {
    public int[] constructArr(int[] a) {
        int[] b = new int[a.length];
        for(int i=0;i<a.length;i++)
        {
            b[i] = 1;
        }
    //两次遍历 一次从左到右，一次从右往左 保证乘到所有的东西
    int left = 1;
    for(int i=0;i<a.length;i++)
    {
        b[i] = left;//这样保证b[i]每次都取到0~i-1
        left *= a[i];
    }
    int right = 1;
    for(int j=a.length-1;j>=0;j--)
    {
        b[j] *= right;//之前的b[i]只计算了左半部分，还需要和右边的合并
        right*= a[j];
    }
    return b;
    }
}
```
## 堆 
### 面试题40. 最小的k个数
输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
```
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if(arr.length==0) return new int[0];
        PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>(){
            @Override
            public int compare(Integer a,Integer b){
            return b-a;//降序
            }

        });
        for(int a:arr)
        {
            queue.add(a);
            if(queue.size()>k) queue.poll();
        }
        int[] res = new int[k];
        for(int i=0;i<k;i++)
        {
            res[i] = queue.poll();
        }
        return res;

    }
}
```
### 41 数据流中的中位数
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
```
class MedianFinder {
    private PriorityQueue<Integer> maxqueue;
    private PriorityQueue<Integer> minqueue;
    private int size;
    /** initialize your data structure here. */
    public MedianFinder() {
        maxqueue = new PriorityQueue<Integer>((x,y)->y-x); //优先队列默认升序 最小堆 最大堆
        minqueue = new PriorityQueue<Integer>();
        size = 0;
    }
    
    public void addNum(int num) {
        size++;
        maxqueue.add(num);
        minqueue.add(maxqueue.poll());
        if((size&1)==1)//奇数
        maxqueue.add(minqueue.poll());

    }
    
    public double findMedian() {
        if((size&1)==1)
        return (double) maxqueue.peek();
        else
        return (double) (maxqueue.peek()+minqueue.peek())/2;
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```
## 排序部分
#### 面试题45. 把数组排成最小的数
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
```
class Solution {
    public String minNumber(int[] nums) {
        if(nums.length==0) return "";
        ArrayList<Integer> list = new ArrayList<>();
        for(int i=0;i<nums.length;i++)
        {
            list.add(nums[i]);
        }
        //需要自定义比较方法，也可以使用Arrays.sort()方法
        Collections.sort(list,new Comparator<Integer>(){
            public int compare(Integer a,Integer b)
            {
                String x = a+""+b;
                String y = b+""+a;
                return x.compareTo(y);//注意类比之前的排序规则，记住comapre（）方法 a-b 默认升序 b-a 降序的区别

            }
        });
        StringBuilder res = new StringBuilder();
        for(int str:list)
        {
            res.append(str);
        }
        return res.toString();
    }
}
```
## 位操作
### 面试题15. 二进制中1的个数
请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。
```
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        //位操作的用法 n&(n-1) 会消除末尾最后一个1，直到n为0即可
        if(n==0) return 0;
        int count = 0;
        while(n!=0)
        {
            count++;
            n = n&(n-1);
        }
        return count;
    }
}
```
### 面试题56 - I. 数组中数字出现的次数
一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
需要总结：
n&(n-1)  将末尾最后一个1转为0
n&(-n) 只保留最后一个1  计算方式 注意计算机按照补码方式计算  需要计算-n的补码 然后与n的补码进行&
n&1 可以判断奇偶  为0 偶数  为1 奇数 
```
class Solution {
    public int[] singleNumbers(int[] nums) {
        int sum = 0;
       //异或可以得到最后两个不同的数之间的和，然后需要找到区分的办法，利用n&(-n)可以保留最后一个1，则说明此位置两者之间肯定不同，可以区分
        for(int i:nums)
        {
            sum ^=i;
        }
        int temp = sum&(-sum); //保留最后一个1的位置，将数组分为两部分
        int [] res = new int[2];
        for(int i: nums)
        {
            if((i&temp)==temp)//此处不能用1区分，因为虽然n&(-n)保留了最后一个1所在的位置，但并不一定在最后一位，若某个数此位置为1，求&之后得到的就是temp. 
            //可以用temp进行区分，也可以用0进行区分。（0的话说明此位置不是1，那么&就全部为0）
            {
                res[0]^=i;
            }
            else 
            {
                res[1]^=i;
            }
        }
        return res;
    }
}
```
### 面试题56 - II. 数组中数字出现的次数 II
在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
```
class Solution {
    public int singleNumber(int[] nums) {
    //按位运算  整理每一位  对3取模则为剩下的值  所有位加起来则为res
        int res = 0;
        for(int i=0;i<32;i++)
        {
            int count = 0;//每一位清零
            for(int t:nums)
            {
                if(((t>>i)&1)==1) count++;
            }
            res += (count%3)<<i;
        }
        return res;
    }
}
```
## 哈希表
###  面试题50. 第一个只出现一次的字符
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。
```
class Solution {
    public char firstUniqChar(String s) {
        if(s==null||s.length()==0) return ' ';
        HashMap<Character,Integer> map = new HashMap<>();
        char res = ' ';
        for(int i=0;i<s.length();i++)
        {
            if(map.containsKey(s.charAt(i)))
            map.put(s.charAt(i),map.get(s.charAt(i))+1);
            else
            map.put(s.charAt(i),1);
        }
         for(int i=0;i<s.length();i++)
         {
             if(map.get(s.charAt(i))==1)
             {
                 res = s.charAt(i);
             break;
             }
         }
         return res;
    }
}
```
## 链表
### 面试题18. 删除链表的节点
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        //引入假头 
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode first = pre;
        ListNode second = head;
        while(true)
        {
            if(second.val==val)
            {
                first.next = second.next;
                break;     
            }
            else
            {
                first = first.next;
                second = second.next; 
            }
        }
        return pre.next; 
    }
}
```
### 面试题22. 链表中倒数第k个节点
输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        if(head==null) return new ListNode(0);
        Stack<ListNode> stack = new Stack<>();
        ListNode pre = head;
        while(pre!=null)
        {
            stack.push(pre);
            pre = pre.next;
        }
        while(k>1)
        {

            stack.pop();
            k--;    
        }
        return  stack.peek();   
    }
}
```
### 面试题52. 两个链表的第一个公共节点
输入两个链表，找出它们的第一个公共节点。  
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
 //注意双指针法的应用  有的话联合遍历一遍总会相遇
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
         if (headA == null || headB == null) return null;
        ListNode node1 = headA;
        ListNode node2 = headB;
        while(node1!=node2)
        {
            node1 = (node1 == null )? headB : node1.next;
            node2 = (node2 == null )? headA : node2.next;
        }
        return node1;
    }
}
```
### 复杂链表的复制
请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

```
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/ 利用hash表完成 复制工作 
class Solution {
    public Node copyRandomList(Node head) {
        if(head==null) return head;
        HashMap<Node,Node> hashmap = new HashMap<>();//旧节点  新节点
        Node node1 = head;
        while(node1!=null)
        {
            hashmap.put(node1,new Node(node1.val));
           node1 = node1.next;
        }
        node1 = head;
        while(node1!=null)
        {
           Node newNode = hashmap.get(node1);
           newNode.next = hashmap.get(node1.next);
           newNode.random = hashmap.get(node1.random);
           node1 = node1.next;
        }
        return hashmap.get(head);
    }
}
```
### 面试题06. 从尾到头打印链表
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] reversePrint(ListNode head) {
        if(head==null) return new int[0];
        ListNode pre = head;
        Stack<Integer> stack = new Stack<>();
        ArrayList<Integer> arr = new ArrayList<>();
        while(pre!=null)
        {
            stack.push(pre.val);
            pre = pre.next;
        }
        while(!stack.isEmpty())
        {
            arr.add(stack.pop());
        }
        int[] res = new int[arr.size()];
        for(int i=0;i<arr.size();i++)
        {
            res[i] = arr.get(i);
        }
        return res;
    }
}
```
### 面试题25. 合并两个排序的链表
输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1==null) return l2;
        if(l2==null) return l1;
        ListNode node = new ListNode(0);
        ListNode res = node;
        ListNode p = l1;
        ListNode q = l2;
        while(p!=null&&q!=null)
        {
            if(p.val<q.val)
            {
                res.next = p;
                p = p.next;
                res = res.next;
            }
            else
            {
                res.next = q;
                q = q.next;
                res = res.next;
            }
        }
        res.next = p==null?q:p;
        return node.next;
    }
}
```
## 二叉树
### 面试题55 - I. 二叉树的深度
输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 [3,9,20,null,null,15,7]，
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null) return 0;
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left,right)+1;
    }
}
```
### 面试题55 - II. 平衡二叉树
输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isBalanced(TreeNode root) {
        //因为深度是int 所以做如此转换
        return depth(root)>=0;
    }
    //求深度的变形  在求深度的同时 比较一下左右子树的高度差即可 
    public int depth(TreeNode root){
        if(root==null)
        return 0;
        int l = depth(root.left);
        int r = depth(root.right);
        if(l<0||r<0)
        return -1;
        if(Math.abs(l-r)>1)
        return -1;
        return Math.max(l,r)+1;
    }
}
**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    boolean flag = true;
    public boolean isBalanced(TreeNode root) {
        if(root==null){
            return true;
        }
        height(root);
        return flag;
    }

    private int height(TreeNode root){
        if(root==null){
            return 0;
        }
        int left = height(root.left);
        int right = height(root.right);
        if(Math.abs(left-right)>1){
            flag = false;
        }
        return 1+Math.max(left, right);
    }
}
```
###  面试题27. 二叉树的镜像

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root==null) return null;//递归出口
        //交换左右子树
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        //递归解决左子树和右子树
        mirrorTree(root.left);
        mirrorTree(root.right);
        return root;
    }
}
```
### 面试题28. 对称的二叉树
请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。 
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
// 递归出口 如果左子树和右子树是空的 返回true  若有一个为空一个不为空  或者两个不相等 则返回false  当左子树与右子树都是对称的 则二叉树是对称的
    public boolean isSymmetric(TreeNode root) {
        boolean res = true;
        if(root==null) return res;
        res = helper(root.left,root.right);
        return res;
    }
    public boolean helper(TreeNode left,TreeNode right)
    {
        if(left==null&&right==null) return true;
        if((left==null&&right!=null)||(left!=null&&right==null))
        return false;
        if(left.val!=right.val)
        return false;
        return helper(left.left,right.right)&& helper(left.right,right.left);
    }
}
```
### 面试题32 - I. 从上到下打印二叉树
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        ArrayList<Integer> arr = new ArrayList<>();
        if(root==null) return new int[0];
        queue.offer(root);
        while(!queue.isEmpty())
        {
            int cnt = queue.size();
            while(cnt-->0)
            {
                TreeNode temp = queue.poll();
                arr.add(temp.val);
                if(temp.left!=null)
                queue.offer(temp.left); 
                if(temp.right!=null)
                queue.offer(temp.right);
            }
        }
        int [] res = new int[arr.size()];
        for(int i=0;i<arr.size();i++)
        {
            res[i] = arr.get(i);
        }
        return res;
    }
}
```
### 面试题32 - II. 从上到下打印二叉树 II
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

 
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if(root==null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty())
        {
            int size = queue.size();//每层的节点个数
            List<Integer> temp = new ArrayList<>();//存放临时变量
            for(int i=0;i<size;i++)
            {
                TreeNode node = queue.poll();
                temp.add(node.val);
                if(node.left!=null)
                queue.add(node.left);
                if(node.right!=null)
                queue.add(node.right);
            }
            res.add(temp);//存放此层的数据
        }
        return res;
    }
}
```
### 面试题32 - III. 从上到下打印二叉树 III
请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>>  res = new ArrayList<>();
        if(root==null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        boolean flag = true;//标记行数
        while(!queue.isEmpty())
        {
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            
            for(int i=0;i<size;i++)
            {
                TreeNode node = queue.poll();
                list.add(node.val);
                if(node.left!=null)
                queue.add(node.left);
                if(node.right!=null)
                queue.add(node.right);
            }
            if(flag==true)
            {
            res.add(list);
            flag = false;
            }
            else{
        Collections.reverse(list);
            res.add(list);
            flag = true;
            }
        }
        return res;
    }
}

```
### 面试题54. 二叉搜索树的第k大节点
给定一棵二叉搜索树，请找出其中第k大的节点。

注意中序遍历的特点，注意递归中对于变量的共享。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    int count ;
    int result = -1;
    public int kthLargest(TreeNode root, int k) {
        count = k;
        helper(root);
        return result;
    }
    //中序遍历二叉搜索树 先左后右 增序  先右后左 降序
    public void helper(TreeNode root)
    {
        if(root.right!=null)
        helper(root.right);
        if(count==1)
        {
            result = root.val;
            count--;
            return;
        }
        count--;
          if(root.left!=null)
        helper(root.left);
    }
}
```
### 面试题68 - I. 二叉搜索树的最近公共祖先
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
 递归版
 class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val<root.val && q.val<root.val){
            return lowestCommonAncestor(root.left,p,q);
        }else if (p.val>root.val && q.val>root.val){
            return lowestCommonAncestor(root.right,p,q);
        }
        return root;
    }
}
迭代版
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        while(root!=null)
        {
            if((p.val<root.val)&&(q.val<root.val))
            root = root.left;
            else if((p.val>root.val)&&(q.val>root.val))
            root = root.right;
            else 
            break;
        }
        return root;
    }
}
```
### 面试题68 - II. 二叉树的最近公共祖先
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    //当root为null  说明结束了 当root为p/q的时候 返回即可 因为自己可以作为节点
        if(root==null||root==p||root==q)
        {
            return root;
        }
	//左右子树的返回结果作为判断
	 
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        //若某处为空 则说明为另一边子树的结果 若都为空 则说明root为祖先
	if(left==null) return right;
        if(right==null) return left;
        return root;
    }
}
```
###  面试题07. 重建二叉树
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if((preorder.length==0)||preorder==null) return null;
        return helper(preorder,0,preorder.length-1,inorder,0,inorder.length-1);
    }
    public TreeNode helper(int[] preorder,int a1,int a2,int[] inorder,int b1,int b2)
    {
        TreeNode root = new TreeNode(preorder[a1]);//此时的根节点
        int i = b1;//从inorder中寻找前序的第一个节点位置
        while(preorder[a1]!=inorder[i])i++; //找到inorder中的值
        int left = i-b1;//按照i点将中序结果分为两半 左边为左子树 右边为右子树
        int right = b2-i;
        if(left>0)
        root.left = helper(preorder,a1+1,a1+left,inorder,b1,i-1);//给左子树递归求值 注意preorder需要去掉a1，inorder需要去掉i元素。
        if(right>0)
        root.right = helper(preorder,a1+1+left,a2,inorder,i+1,b2);
        return root;
    }
}
```
### 面试题36. 二叉搜索树与双向链表
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
```
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    public Node pre = null;
    public Node treeToDoublyList(Node root) {
        //中序遍历 利用pre节点存储前一个节点 若此节点不为空 则将root的left也就是前一个节点的right指针变为root,此后pre指向root 继续进行此操作。      
        if(root==null) return null;
          Node head=root,tail=root;
        while(head.left!=null) head = head.left;  //寻找最小的节点
        while(tail.right!=null) tail= tail.right; //寻找最大的节点
        helper(root);
        head.left = tail;
        tail.right = head;
        return head;
    }
    public void helper(Node root)
    {
        if(root==null) return;
        helper(root.left);
        //中序遍历
        root.left = pre;
        if(pre!=null)
        pre.right = root;
        pre = root;
        helper(root.right);
    }
}
```
### 面试题34. 二叉树中和为某一值的路径
输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
    if(root==null) return res;
    helper(root,sum,new ArrayList<>());
    return res;
    }
    public void helper(TreeNode node,int sum,List<Integer> list)
    {
        list.add(node.val);//添加节点值进入list
        if((sum-node.val==0)&&(node.left==null)&&(node.right==null))
        res.add(new ArrayList(list));
        else
        {
            if(node.left!=null)
            {
                helper(node.left,sum-node.val,list);
                list.remove(list.size()-1);
            }
            if(node.right!=null)
            {
                helper(node.right,sum-node.val,list);
                list.remove(list.size()-1);
            }

        }
    }
}

```
### 面试题33. 二叉搜索树的后序遍历序列
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 
```
class Solution {
    //后序遍历最后一个一定是根
    //二叉搜索数左子树比根小 右子树比根大
    public boolean verifyPostorder(int[] postorder) {
        if(postorder==null){
            return false;
        }
        return partitioin(postorder, 0, postorder.length-1);
    }

    public boolean partitioin(int[] postorder, int left, int right){
        if(left>=right){
            return true;
        }
        int pivot = postorder[right];
        int index = right-1;
        while(index>=left && postorder[index]>pivot){
            index--;
        }
        boolean isAllLess = true;
        for(int i=index;i>=left;i--){
            if(postorder[i]>pivot){
                isAllLess = false;
                break;
            }
        }
        if(isAllLess==false){
            return isAllLess;
        }else{
            return partitioin(postorder, left, index) && partitioin(postorder, index+1, right-1);
        }
    }
}
```
### 26 面试题26. 树的子结构
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(A==null||B==null) return false;
        return iscur(A,B)||isSubStructure(A.left,B)||isSubStructure(A.right,B);
    }
    boolean iscur(TreeNode a,TreeNode b)
    {
        if(b==null) return true;
        if(a==null|| a.val!=b.val) return false;
        return  iscur(a.left,b.left) && iscur(a.right,b.right);
    }
}
```
## 数学部分
### 面试题12. 矩阵中的路径
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

```
class Solution {
    public boolean exist(char[][] board, String word) {
//dfs+回溯法
    if(board.length==0||board[0].length==0||word.length()==0)
    return false;
    boolean[][] flags = new boolean[board.length][board[0].length];
    for(int i=0;i<board.length;i++)
    {
        for(int j=0;j<board[0].length;j++)
        {
            if(find(board,i,j,flags,0,word))
            return true;
        }
    }
    return false;
    }
    public boolean find(char[][]board,int i,int j,boolean[][] flags,int count,String word)
    {
        //退出条件
        if(count==word.length())
        return true;
        if(i>=0&&i<board.length && j>=0 && j<board[0].length && flags[i][j]==false && board[i][j]==word.charAt(count))
        {
        count++;
        flags[i][j] = true;//标记来过了
        if(find(board,i-1,j,flags,count,word)||find(board,i+1,j,flags,count,word)||find(board,i,j-1,flags,count,word)||find(board,i,j+1,flags,count,word))
        return true;
        else
        {
            count--;
            flags[i][j] = false;
        }
        }
        return false;
    }
}
```
### 面试题13. 机器人的运动范围
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

```
class Solution {
    private int res=0,k,m,n;//res设为全局变量方便递归更改 dfs很简单 就一条走到黑就行。
    
    private boolean[][] move; 
    public int movingCount(int m, int n, int k) {
        this.k = k;
        this.m = m;
        this.n = n;
        move = new boolean[m][n];
        dfs(0,0);
        return res;
    }
    public void  dfs(int i,int j)
    {
        if(i>=0 && i<m && j>=0 && j<n && caculate(i)+caculate(j)<=k && !move[i][j])
        {
            res++;
            move[i][j] = true;
            dfs(i-1,j);
            dfs(i+1,j);
            dfs(i,j-1);
            dfs(i,j+1);
        }
    }
    //要记住 %10 可以拿到最后一位数字 /10可以拿到除去最后一位数字的部分
    public int caculate(int num)
    {
        int tmp = 0;
        while(num>0)
        {
            tmp += num%10;
            num = num/10;
        }
        return tmp;
    }
}
```
### 14 剪绳子I
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
```
//dp算法
class Solution {
    public int cuttingRope(int n) {
        int[] dp = new int[n+1];
        dp[1] = 1;
        dp[2] = 1;
        for(int i=3;i<=n;i++)
        {
            for(int j=1;j<i;j++)//dp[i] 表示此段绳子的最大值
            dp[i] = Math.max(dp[i],Math.max(dp[i-j]*j,(i-j)*j));
        }
        return dp[n];
    }
}
```
### 14 剪绳子II
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

```
//贪心算法  一步步分析 最后分为3和2即可获得最大值。
class Solution {
    public int cuttingRope(int n) {
        if(n<4) return n-1;
        long res = 1;
        while(n>4)
{
        n-=3;
        res = (res*3)%1000000007;
}
        return (int)((n*res)%1000000007);
        }
}

```
### 面试题20. 表示数值的字符串
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"0123"及"-1E-16"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。

```
class Solution {
    public boolean isNumber(String s) {
        //核心问题是分开讨论 1.讨论数字类型（包括小数点） 2.讨论e指数问题
        //清楚空格  注意放在第一步，不然如果trim之后为空 会出现out of range的错误
        s = s.trim(); 
        if(s.length()==0||s==null) return false;
        int cnt = 0;//用做统计

        //判断符号
        if(s.charAt(cnt)=='+'||s.charAt(cnt)=='-')
        {cnt++;}
        int numcnt = 0;
        int point = 0;
        //讨论数字类型
        while(cnt!=s.length()&&((s.charAt(cnt)>='0'&&s.charAt(cnt)<='9')||s.charAt(cnt)=='.'))
        {
            if(s.charAt(cnt)=='.') 
            point++;
            else
            {numcnt++;}
            cnt++;
        }
        //判断数字合法情况
        if(point>1||numcnt<1)
        return false;
        if(cnt==s.length())
        return true;
        //判断e指数问题
        if(s.charAt(cnt)=='e') //后面的判断都需要在s.charAt(cnt)=='e'的条件下
       { cnt++;
        if(cnt==s.length()) return false;
        if(s.charAt(cnt)=='-'||s.charAt(cnt)=='+')
        cnt++;
        if(cnt==s.length()) return false;
        while(cnt!=s.length()&&((s.charAt(cnt)>='0'&&s.charAt(cnt)<='9')))
        {
            cnt++;
        }
         if(cnt==s.length()) return true;
         }
         return false;//其他各种情况 
    }
}
```
### 10 面试题10- I. 斐波那契数列
```
//dp算法
class Solution {
    public int fib(int n) {
        if(n==0) return 0;
        if(n==1) return 1;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        for(int i=2;i<=n;i++)
{
    dp[i] = (dp[i-1]+dp[i-2])%1000000007;
}
return dp[n];
    }
}
//
class Solution {
    public int fib(int n) {
        if(n==0) return 0;
        if(n==1) return 1;
        int num1 = 0;
        int num2 = 1;
        int res = 0;
        for(int i=2;i<=n;i++)
{
    res = (num1+num2)%1000000007;
    num1 = num2;
    num2 = res;
}
return res;
    }
}
```
### 面试题10- II. 青蛙跳台阶问题
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

```
class Solution {
    public int numWays(int n) {
        if(n==1) return 1;
        if(n==0) return 1;
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i=2;i<=n;i++)
        {
            dp[i] = (dp[i-1]+dp[i-2])%1000000007;
        }
        return dp[n];
    }
}
```
### 面试题42. 连续子数组的最大和
输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。
```
class Solution {
    public int maxSubArray(int[] nums) {
    int temp = nums[0];
    int res =  nums[0];
        for(int i=1;i<nums.length;i++)
        {
            temp = Math.max(temp+nums[i],nums[i]);
            res = Math.max(temp,res);
        }
        return res;
    }
}
```
### 面试题57. 和为s的两个数字
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        if(nums.length==1) return new int[0];
        int [] res = new int[2];
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<nums.length;i++)
        {
            if(!map.containsKey(nums[i]))
            {
                map.put(nums[i],i);
                if(map.containsKey(target-nums[i])&&map.get(target-nums[i])!=i)
                {
                    res[0] = nums[i];
                    res[1] = target-nums[i];
                    break;
                }
            }
        }
        return res;
    }
}
```
### 面试题57 - II. 和为s的连续正数序列
输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
```
class Solution {
    public int[][] findContinuousSequence(int target) {
        //利用双指针
       List<int[]> res = new ArrayList<>();
        int left = 1;
        int right = 2;
        int count = left+right;
        while(left<right&&left<(target+1)/2)
        {
            if(count<target)
            {
                right++;
                count +=right;
            }
            else if(count>target)
            {
                count -= left;
                left++;
            }
            else
            {
                int [] temp = new int[right-left+1];
                int j = left;
                for(int i=0;i<right-left+1;i++,j++)
                {
                    temp[i] = j;
                }
                res.add(temp);
                //加入之后看后面的
                count -= left;
                left++;
            }
        }
     int[][] out = new int[res.size()][];
     return res.toArray(out);
    }
}
```
### 面试题60. n个骰子的点数
把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。
```
class Solution {
    public double[] twoSum(int n) {
        //本题目本质是DP相关问题
        //n个骰子可以取得数值为n-6*n
        //递推关系式为dp[n][j] = 1/6(dp[n-1][j-1]+dp[n-1][j-2]+...+dp[n-1][j-6])
        //dp[n]只与dp[n-1]的关系，假设此时和为j,dp[n][j]只能取1-6,概率分别为1/6,当取1的时候应该考虑dp[n-1][j-1],取2的时候应该考虑dp[n-1][j-2]，等等，全完之后则就是dp[n][j]
        double[][] dp = new double[n+1][6*n+1];
        double p = 1/6.0;
        for(int i=1;i<=6;i++)
        dp[1][i] = p;
        //色子数
        for(int i=2;i<=n;i++)
        {
            //计算概率,j的取值为i-6*n
            for(int j=i;j<=6*n;j++)
            {
                for(int k=1;j-k>=0&&k<=6;k++)
                dp[i][j] +=p*dp[i-1][j-k];
            }
        }
        ArrayList<Double> res = new ArrayList<>();
            for(int i=n;i<=6*n;i++)
        {
            res.add(dp[n][i]);
        }
      double[] result = new double[res.size()];
      for(int i=0;i<res.size();i++)
      {
          result[i] = res.get(i);
      }
      return result;
    }
}
```
### 面试题47. 礼物的最大价值

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

```
class Solution {
    public int maxValue(int[][] grid) {
        if(grid==null) return 0;
        int row = grid.length;
        int col = grid[0].length;
        //更新边界值
        for(int i=1;i<row;i++)
        {
            grid[i][0] += grid[i-1][0]; 
        }
        for(int i=1;i<col;i++)
        {
            grid[0][i] += grid[0][i-1]; 
        }
        for(int i=1;i<row;i++)
        {
            for(int j=1;j<col;j++)
            {
                grid[i][j] += Math.max(grid[i-1][j],grid[i][j-1]);
            }
        }
        return grid[row-1][col-1];
    }
}
```
### 面试题63. 股票的最大利润
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
```
class Solution {
    public int maxProfit(int[] prices) {
        //dp dp[i] = dp[i-1] + price[i]-price[i-1] 
        //今天的最大利润等于昨天的最大利润+昨天与今天的差价
        if(prices.length==0) return 0;
        int [] dp = new int[prices.length];
        int result = 0;
        dp[0] = 0;
        for(int i=1;i<dp.length;i++)
        {
            dp[i] = dp[i-1]+prices[i]-prices[i-1];
            dp[i] = dp[i]>0?dp[i]:0;
            result = Math.max(result,dp[i]);
        }
        return result;
    }
}
```
### 面试题49. 丑数
我们把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
```
class Solution {
    public int nthUglyNumber(int n) {
        int p2=0,p3=0,p5=0;
        //利用dp解决
        if(n==1) return 1;
        int[] dp = new int[n];
        dp[0] = 1;
        for(int i=1;i<n;i++)
        {
            dp[i] = Math.min(dp[p2]*2,Math.min(dp[p3]*3,dp[p5]*5));
            if(dp[i]==dp[p2]*2) p2++;
            if(dp[i]==dp[p3]*3) p3++;
            if(dp[i]==dp[p5]*5) p5++;
        }
        return dp[n-1];
    }
}
```
### 面试题46. 把数字翻译成字符串
给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

```
class Solution {
    public int translateNum(int num) {
        if(num<=9) return 1;
        String s = String.valueOf(num);
        int len = s.length();
        int[] dp = new int[len+1];
        dp[0] = 1; //从1开始计算dp  但是访问数组的时候需要下标减一
        for(int i=1;i<=len;i++)
        {
            dp[i] +=dp[i-1];
            if(i>1)
            {
           if(s.charAt(i-2)=='1'||(s.charAt(i-2)=='2'&&s.charAt(i-1)<'6'))
                dp[i] +=dp[i-2];
            }
        }
        return dp[len];
    }
}
```
### 面试题38. 字符串的排列
输入一个字符串，打印出该字符串中字符的所有排列。

 

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
```
//本质是全排列问题 第一个数与后面所有一一交换位置， 后面的根据递归采取同样的办法即可 hashse用于去重
class Solution {
   public  HashSet<String> set = new HashSet<>(); //用于去重
    public String[] permutation(String s) {
        //全排列问题就是递归问题
        if(s==null) return new String[0];
        char [] ch = s.toCharArray();
        helper(ch,0);
        return set.toArray(new String[0]);
            }
        public void helper(char[] s,int begin)
        {
            if(begin==s.length-1)
            {
                set.add(new String(s));
                return;
            }
            for(int i=0;i<s.length;i++)
            {
                swap(s,begin,i);
                helper(s,begin+1);
                swap(s,i,begin);
            }
        }
        public void swap(char[] s,int l,int r)
        {
            char temp = s[l];
            s[l] = s[r];
            s[r] = temp;
        }
}
```
# Leetcode笔记 
## 数组部分

### 两数之和   返回和为target的两个元素的下标
#### 方法一：暴力法  时间复杂度o(n2)  没有多余开辟的空间
```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[j] == target - nums[i]) {
                    return new int[] { i, j };
                }
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}
```
#### 方法二：hash map 
* 利用hash以空间换取时间的思想 时间复杂度为o(n) 但是要考虑如果hash冲突的问题  
* 可以从这个角度考虑 不要先将所有的值都加入hash map中去，每个元素加进去的必要条件是当前这个元素和hash map中的元素没有sum为target的值，如果有的话直接返回（即如果是两个下标不同，但value相同的元素需要采取此办法）。
```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[] { map.get(complement), i };
            }
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}
```
### 15 三数之和
#### 利用双指针和排序算法完成 注意去重的部分  O(n2)
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList();
        if(nums.length<=2) return res;
        //快排
        Arrays.sort(nums);
        for(int i=0;i<nums.length;i++)
        {
            if(nums[i]>0) break;//加快收敛速度 若此处大于0 则不会有相加为0的情况
            if(i>0 && nums[i]==nums[i-1]) continue;//去重
            int left = i+1;
            int right = nums.length-1;
            while(left<right)
            { 
             int sum = nums[i]+nums[left]+nums[right];
            if(sum==0)
            {
                List<Integer> temp = new ArrayList();
                temp.add(nums[i]);
                temp.add(nums[left]);
                temp.add(nums[right]);
                res.add(temp);
                //去重 
                while(left<right && nums[left+1]==nums[left]) left++;
                while(left<right && nums[right-1]==nums[right]) right--;
                right--;
                left++;
            }
            else if(sum<0)
                left++;
             else
                 right--;
             }
        }
        return res;
}
}

```
### 6 最长回文字串
#### 回文数：中心对称的字符串 
	利用动态规划算法，dp[i][j] = dp[i+1][j-1]&& s[i]==s[j] 利用二维数组存取所有的[i][j]取值情况，然后若为true,看字符串长度是多少，然后更新最长回文子串，最终就会找到结果。
```
	class Solution {

public String longestPalindrome(String s) {

        if (s.length() < 2) { // 单个字符肯定是回文串，直接返回s
            return s;
        }
        boolean[][] dp = new boolean[s.length()][s.length()];  // 初始化一个二维数组，值默认是false
        String result = s.substring(0,1);
        for (int j = 0; j < s.length(); j++){
            for (int i = 0; i <= j; i++){
                if((j-i)<=2){
                    dp[i][j] = s.charAt(i) == s.charAt(j);
                }
                else
                dp[i][j] = s.charAt(i) == s.charAt(j) && dp[i+1][j-1];
                if (dp[i][j]){
                    if (j - i + 1 > result.length()){
                        result = s.substring(i, j + 1);
                    }
                }
            }
        }
        return result;
}}
```
### 7 整数反转
#### 带有溢出的反转 考虑int溢出的问题 注意整数的表示
```
class Solution {
    public int reverse(int x) {
        long res = 0;
        while(x!=0)
        {
            int pop = x%10;
            res =res*10 + pop;
            if(res>Integer.MAX_VALUE||res<Integer.MIN_VALUE)
                return 0;
            x/=10;
        }
        return (int)res;
    }
}
```
### 8 字符串转换 atio
#### 常规做法  遍历一遍字符串，把合格的字符串部分转换为数字。 递增一次检查是否越界
```
import java.util.regex.*;
class Solution {
    public int myAtoi(String str) {
        str = str.trim();//清除字符串前后空格
        if(str.equals(" ") || str.length()==0)
            return 0;
        int start = 0; //用于标识数字字符串开始的位置，因为可能有+或者-符号
        int flag = 1; //标识+-
        long res = 0;  //防止越界用long存储
        if(str.charAt(0)=='+')
            start = 1;
        if(str.charAt(0)=='-')
        {start = 1;
            flag = -1;}
        for(int i = start;i<str.length();i++)
        {
            if(str.charAt(i)<'0'||str.charAt(i)>'9')
                return (int)res*flag;
            else
            {
                res = res*10+(int)(str.charAt(i)-'0');
                if(flag==1 && res>Integer.MAX_VALUE)
                { res = Integer.MAX_VALUE;
                    break;}
                if(flag==-1 && res>Integer.MAX_VALUE)
                { res = Integer.MIN_VALUE;
                    break;}
                
            }
          }
        return (int)res*flag;
        
    }
}	
		
```
#### 利用正则表达式进行匹配 ^[\\+\\-]?\\d+  ^匹配字符串开头 [] 里面任意一个均可 \\+ \\-代表正负号 ？代表前面的可有可无 \\d代表数字 +代表多个   注意m.end()返回匹配最后一位的后一位，substring（a，b）返回a b-1 
```
import java.util.regex.*;
class Solution {
    public int myAtoi(String str) {
        //清空字符串开头和末尾空格（这是trim方法功能，事实上我们只需清空开头空格）
        str = str.trim();
        //java正则表达式
        Pattern p = Pattern.compile("^[\\+\\-]?\\d+");
        Matcher m = p.matcher(str);
        int value = 0;
        //判断是否能匹配
        if (m.find()){
            //字符串转整数，溢出
            try {
                value = Integer.parseInt(str.substring(m.start(), m.end()));
            } catch (Exception e){
                //由于有的字符串"42"没有正号，所以我们判断'-'
                value = str.charAt(0) == '-' ? Integer.MIN_VALUE: Integer.MAX_VALUE;
            }
        }
        return value;
    }
}

```
### 2 两数相加
#### 此题主要问题在于如何处理相加的问题，同时要生成新的链表，考虑小学学过的加法，设置进位标志`carry`,长度不一的只需在高位补0即可，最后一定要注意最后一位的carry是否会有进位。  此外，如果不设置头结点（无用，只是用来标记头部），则首节点设置会非常麻烦。
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode res = new ListNode(0);
    ListNode p = l1, q =l2, curr = res;
        int carry = 0;
        while(p!=null||q!=null){
          int x = (p!=null)?p.val: 0;
          int y = (q!=null)?q.val: 0;
          int temp = x+y+carry;
          carry = temp/10; 
          curr.next = new ListNode(temp%10);
            //向后遍历
            curr = curr.next;
          if(p!=null) p = p.next;
          if(q!=null) q = q.next;
        }
        if(carry==1)
            curr.next = new ListNode(1);
        return res.next;
    }
}
```
### 3 最长不重复子串
#### way2:利用滑动窗口，利用set集合保存字符，首先寻找从第一个元素开始的最长的子串，找到后记下来更新为max,之后去掉此字符，从第二个开始，但注意此时不需要重新开始设置set集合，此时j指针指向的位置肯定和前面某一个字符重复，虽然无法确定，但可以继续执行，若和去掉的i重复，则j可以依次向后，若不是，i会继续加大，找到重复的那个元素，从而继续加大j，直到j到达末尾，则max就是最大的子串长度，此方法只需要o(2n）的复杂度。
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if(s.length()==0) return 0;
        int max = 0,j = 0,i = 0;
        int n = s.length();
        Set<Character> m = new HashSet<Character>();
        while(i<n&&j<n)
        {
            if(!m.contains(s.charAt(j)))
            {
                m.add(s.charAt(j));
                max = Math.max(max,j-i+1);
                j++;
            }
            else
                m.remove(s.charAt(i++));//无论是否与第i个元素重复，此时都需要删掉此元素，因为后面要找的子问题不包括此元素。
          }
        return max;
    }
}
```
### 11盛最多水的容器
#### way1 暴力法
```
class Solution {
    public int maxArea(int[] height) {
        int max = 0;
        for(int i=0;i<height.length;i++)
        {
            for(int j=i+1;j<height.length;j++)
            {
                int temp = Math.min(height[i],height[j]);
                temp = temp*(j-i);
                if(temp>max) max = temp;
}
        }
        return max;
    }
}

```
#### way 2 双指针法  面积取决于坐标轴差距以及最低的边 
```
class Solution {
    public int maxArea(int[] height) {
        int max = 0;
        int left = 0;
        int right = height.length-1;
       while(left<right)
       {
           max = Math.max(max,Math.min(height[left],height[right])*(right-left));
           if(height[left]>height[right])
               right--;
           else
               left++;
       }
        return max;
    }
}
```
### 125 验证回文串
#### 思路很简单 删掉非法字符 转换字符大小写 之后利用双指针
```
class Solution {
    public boolean isPalindrome(String s) {
        if(s.length()==0) return true;
        s =s.toLowerCase();
        String res = new String();
        for(int i=0;i<s.length();i++)
        {
            if((s.charAt(i)>='0'&&s.charAt(i)<='9')||(s.charAt(i)>='a'&&s.charAt(i)<='z'))
            {
                res +=s.charAt(i);
            }
        }
        int left = 0;
        int right = res.length()-1;
        while(left<=right)
        {
            if(res.charAt(left)!=res.charAt(right))
            {
                return false;
            }
            else
            {
                left++;
                right--;
            }
        }
        return true;
    }
}
```
### 29 两数相除
#### 注意利用位运算、异或运算的性质  题目限定不允许使用除法，但我们要思考到除法的本质还是加法的累计，考虑到每次减去一个divisor太少了，复杂度过高，我们采用每次扩大divisor二倍，这样我们可以降低循环的次数，之后如果跨越太大而dividend不够的话我们可以取差值重新开始，直到最后可以得到result。
>注意Java中的Integer.MAX_VALUE Integer.MIN_VALUE Math.abs()  
>注意左移一位是乘2 右移一位是除2 
>注意int类型的取值范围（-2^n,2^n-1) 根据题目所有的特殊情况都输出MAX_VALUE 
```
class Solution {
    public int divide(int dividend, int divisor) {
        if(divisor==0) return Integer.MAX_VALUE;
        if(dividend==Integer.MIN_VALUE&&divisor==-1) return Integer.MAX_VALUE;//注意int有符号数的取值范围
        long dvd = Math.abs((long)dividend);
        long dvs = Math.abs((long) divisor);
        int result = 0;
        boolean flag = true;
        if((dividend<0)^(divisor<0))//符号相同为0 符号不相同为1
            flag = false;
        while(dvd>=dvs){
        long temp = dvs;
        int res = 1;
            while(dvd>=(temp<<1))
            {
                temp = temp<<1;
                res = res<<1;
                }
            result +=res; 
            dvd -=temp;
        } 
        return flag?result:-result;
    }
}
```
### 14 最大公共子字符串
#### 首先找到字符串数组中最短的字符串(主要为了后面数组不越界)，之后利用两层循环找到最短的公共字符串。 	

`时间复杂度o(n2)`  `空间复杂度o(1) `
```
class Solution {
    public String longestCommonPrefix(String[] strs) {
        int length = strs.length;
        String res = "";
        if(length==0) return "";
        int min_length = Integer.MAX_VALUE;
        for (int i = 0; i < strs.length; i++) {
            if (min_length > strs[i].length())
                min_length = strs[i].length();
        }
        for(int j=0;j<min_length;j++)
        { for(int i=1;i<length;i++)
        {
            if(strs[0].charAt(j)!=strs[i].charAt(j))
                 return res;
        }
         res += strs[0].charAt(j);
        }
        return res;
    }
}
```
### 最大连续子序列和
* 这是一个典型的动态规划问题，考虑temp保存当前最大的连续自序和，但要考虑temp保存的恰是以子序列的结束点为基准，所以递推关系式为temp = max(temp+num[i],num[i]) 此处的最大和要么是跟上一个子序列相加得到的，要么就是自己。 之后res用来保存最大的temp即是结果。
```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int temp = nums[0];
        int res = nums[0];
        if(nums.size()==0) return 0;
        for(int i=1;i<nums.size();i++)
        {
            temp = max(temp+nums[i],nums[i]);
            res = max(res,temp);
        }
    return res;
    }
};
```
### 13 罗马数字转为阿拉伯数字
#### 考虑特殊的数字 若下一位比前一位大则需要特殊处理，否则直接相加即可  记得用hash表将映射关系存储。 
`空间复杂度o(n)` `时间复杂度o(n)`
```
class Solution {
    public int romanToInt(String s) {
        int res = 0;
        HashMap<Character,Integer> map = new HashMap<>();
        map.put('I',1);
        map.put('V',5);
        map.put('X',10);
        map.put('L',50);
        map.put('C',100);
        map.put('D',500);
        map.put('M',1000);
        for(int i=0;i<s.length();i++)
        {
            if(i==s.length()-1||(map.get(s.charAt(i))>=map.get(s.charAt(i+1))))
               res +=map.get(s.charAt(i));
            else
            { res += map.get(s.charAt(i+1))-map.get(s.charAt(i));
               i++;
            }
          }
        return res;
         }
    
    }

```
### 34 寻找一个升序数组中一个数第一次和最后一次出现的位置
#### 题目要求二分查找的复杂度，所以在第一次找到之后要继续利用二分查找 找到边界 利用两个函数分别寻找 目的在于逼近最左和最右
```
class Solution {
    public int searchleft(int[] nums,int target){
        int left = 0;
        int right = nums.length-1;
        int mid = 0;
        int res = -1;
        while(left<=right)
        {
            mid = left + (right-left)/2;
            if(nums[mid]>=target)//注意在找到一个值之后将其向左边逼近，继续利用二分查找
                right = mid-1;
            if(nums[mid]<target)
                left = mid+1;
            if(nums[mid]==target)
                res = mid;
        }
        return res;
    }
        public int searchright(int[] nums,int target){
        int left = 0;
        int right = nums.length-1;
        int mid = 0;
        int res = -1;
        while(left<=right)
        {
            mid = left + (right-left)/2;
            if(nums[mid]>target)
                right = mid-1;
            if(nums[mid]<=target) //注意在找到一个值之后将其向右边逼近，继续利用二分查找
                left = mid+1;
            if(nums[mid]==target)
                res = mid;
        }
        return res;
    }
    public int[] searchRange(int[] nums, int target) {
        int [] res = new int[2];
         res[1] = searchright(nums,target);
        res[0] = searchleft(nums,target);
        return res;
        
    }
}
```
### 387 字符串中的第一个唯一字符
#### 第一次遍历，利用hashmap总结所有的字母出现的次数，之后再遍历一次，若value为1，则返回索引即可。 
`时间复杂度o(n)``空间复杂度o(n)`
```
class Solution {
    public int firstUniqChar(String s) {
        if(s.length()==0) return -1;
        if(s.length()==1) return 0;
        HashMap<Character,Integer> map = new HashMap<>();
        for(int i=0;i<s.length();i++)
        {
            if(map.containsKey(s.charAt(i))==false)       
             map.put(s.charAt(i),1);    
            else
            {
                  int val = map.get(s.charAt(i))+1;
                  map.put(s.charAt(i),val); 
        }
    }
                for(int i=0;i<s.length();i++)
                {
                  if( map.get(s.charAt(i))==1)
                      return i;
                }
        return -1;
}
}
```
### 22 括号生成
#### 利用回溯法完成 注意回溯法的套路 找到回溯出口，然后如何加进去构造括号。
	void back(List<String> res,String temp,int left,int right,int nums)
	res：保存最后的结果  temp：某一个结果  left：左括号个数 right:右括号个数  nums:总个数
```
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if(n==0) return res;
        String temp = new String();
        back(res,temp,0,0,n);
        return res;
        
    }
    public void back(List<String> res,String temp,int left,int right,int nums)
    {
    if(temp.length()==nums*2)
    {
        res.add(temp);
        return;
    }
        if(left<nums)
            back(res,temp+"(",left+1,right,nums);
        if(right<left)
            back(res,temp+")",left,right+1,nums);
    }
}
```
### 28 实现strStr()
#### 给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
注意若needle为空，则返回0(java的indexOf())  采用暴力匹配法 
`时间复杂度o((n-m)m)``空间复杂度o(1)`
```
class Solution {
    public int strStr(String haystack, String needle) {
        if(needle.length()==0) return 0;
        int cnt = 0;
        for(int i=0;i<=haystack.length()-needle.length();i++)
        {

            for(int j=0;j<needle.length();j++)
            {
                if(needle.charAt(j)==haystack.charAt(i+j)){
                    cnt++;
                }
            }
            if(cnt==needle.length())
                return i;
            else
                cnt = 0;
        }
        return -1;
    }
}
```
### 买卖股票的最佳时机  (找到最大的山谷和山峰的区间)
* 我们需要找出给定数组中两个数字之间的最大差值（即，最大利润）。此外，第二个数字（卖出价格）必须大于第一个数字（买入价格）。
形式上，对于每组i和 j（其中 j > i）我们需要找出max(prices[j]−prices[i])。
#### 方法一：暴力法
```
public class Solution {
    public int maxProfit(int prices[]) {
        int maxprofit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            for (int j = i + 1; j < prices.length; j++) {
                int profit = prices[j] - prices[i];
                if (profit > maxprofit)
                    maxprofit = profit;
            }
        }
        return maxprofit;
    }
}
```
* 时间复杂度：O(n^2)
* 空间复杂度：O(1)


#### 方法二：一次遍历
* 假设给定的数组为：[7, 1, 5, 3, 6, 4]
* 如果我们在图表上绘制给定数组中的数字，我们将会得到一个函数图
* 使我们感兴趣的点是上图中的峰和谷。我们需要找到最小的谷之后的最大的峰。
* 我们可以维持两个变量——minprice 和 maxprofit，它们分别对应迄今为止所得到的最小的谷值和最大的利润（卖出价格与最低价格之间的最大差值）
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minprof = 99999;
        int maxprof = 0;
        if(prices.size()==0) return 0;
        for(int i=0;i<prices.size();i++)
        {
            if(prices[i]<minprof){
                minprof = prices[i]; //更新最小的节点
            }
            else if(prices[i]-minprof> maxprof)
            {
                maxprof = prices[i]-minprof;
            }
                }
        return maxprof;
    }
};
```
复杂度o(n)
```
方法3：dp
class Solution {
    public int maxProfit(int[] prices) {
        //dp dp[i] = dp[i-1] + price[i]-price[i-1] 
        //今天的最大利润等于昨天的最大利润+昨天与今天的差价
        if(prices.length==0) return 0;
        int [] dp = new int[prices.length];
        int result = 0;
        dp[0] = 0;
        for(int i=1;i<dp.length;i++)
        {
            dp[i] = dp[i-1]+prices[i]-prices[i-1];
            dp[i] = dp[i]>0?dp[i]:0;
            result = Math.max(result,dp[i]);
        }
        return result;
    }
}
```

### 卖股票的最佳时机II（可以多次买卖，让利润最大）
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
一次遍历：复杂度o(n)
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int profit = 0;
        if(prices.size()==0) return 0;
        for(int i=0;i<prices.size()-1;i++)
        {
            if(prices[i]<prices[i+1])
                profit +=prices[i+1]-prices[i];
        }
        return profit;
    }
};
```
### 36 数独
#### 注意构建三张表  分别表示行 列 九宫格 在遍历一次的情况下对三个表进行更新
```
class Solution {
    public boolean isValidSudoku(char[][] board) {
    boolean[][] row = new boolean[9][9];
        boolean[][] col = new boolean[9][9];
        boolean[][] gong = new boolean[9][9];
        for(int i =0;i<9;i++)
        {
            for(int j=0;j<9;j++)
            {
                if(board[i][j]!='.')
                {     int temp = board[i][j]-'1';//转为int  保证0-8的下表取值
                if(row[i][temp]||col[j][temp]||gong[(i/3)*3+j/3][temp]) //注意找三张表分别存储行、列、九宫格中1-9出现的次数
                    //（正好有九个九宫格，i代表行 j代表列 取值均为0-8） i/3为第几块九宫格 又有三行为一个 所以乘以3找到这边的行数 在加上j/3定位列所属的九宫格 一起定位到第几个九宫格，然后更新对应的temp   
                    return false;
                else
                {
                    row[i][temp] = true;
                        col[j][temp]=true;
                        gong[(i/3)*3+j/3][temp]=true;
                }
                 }
            }
        }
    return true;
    }
}
```
### 存在重复元素
如果数组中有出现两次以上的元素，则返回true,否则返回false
#### 方法1：调用sort排序,之后遍历，一旦发现有相同的元素直接return    复杂度o(nlogn)
#### 方法2：set 将元素放入set中，每次放进去的时候进行对比。
```
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
         set <int> hashset;
        if(nums.size()<=1) return false;
        for(int i=0 ;i<nums.size();i++){
            if(hashset.count(nums[i])==1){
                return true;
            }
            else{
                hashset.insert(nums[i]);
            }
        }
        return false;
    }

};
```
#### 方法3：hash表

### 翻转数组
* 考虑利用%进行翻转，翻转k次的结果等于l=k%n次。
* 首先翻转所有的数组，之后翻转l的部分，最后翻转n-l的部分即可
* 时间复杂度o(n) 空间复杂度o(1)

```
class Solution {
public:
  void  reverse(vector<int> &nums, int left, int right)
    {     int temp =0;
        while(left<right)
        {
            temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            left++;
            right--;   
        }
    }
    void rotate(vector<int>& nums, int k) {
        if(k==0) return ;
        k = k%nums.size();
        reverse(nums,0,nums.size()-1);
        reverse(nums,0,k-1);
        reverse(nums,k,nums.size()-1);
    }

};
```
### 9回文数
#### 最简单的方法就是转为字符串 然后利用两个指针从前和后对比 下面是利用数学方法解决  
```
class Solution {
    public boolean isPalindrome(int x) {
        if(x<0) return false;
        if(x==0) return true;
        int div = 1;
        while(x/div>=10) div*=10;//获得首位
        while(x>0)
        {
            int left = x/div;
            int right = x%10;
            if(left!=right) return false;
            x = (x % div) /10; //去掉x的头和尾 保留中间的数字
            div/=100; // 用于下次更新的div应该比之前小100
        }
        return true;
    }
}
```
### 38 报数
#### 注意看规律 之后按照数学归纳法推理就行
```
class Solution {
    public String countAndSay(int n) {
        if(n==1) return "1";
        String temp = "11";
        if(n==2) return temp;
        while(n-2>0){
            StringBuffer str = new StringBuffer();
            char single = temp.charAt(0);
            int cnt=1;
            for(int i=1;i<temp.length();i++)
            {      
            if(temp.charAt(i)==single)
            {
               cnt++;   
            }
            else
            {
                str.append(cnt).append(single);
                single = temp.charAt(i);
                cnt = 1;
            }  
            }
            str.append(cnt).append(single);//注意将结尾的字符串加入stringbuffer中
            temp = str.toString();//每次将生成的str转为string作为新的temp从而生成下一个数字对应的字符串 最终回生成第n个
            n--;
        }
        return temp;
    }
}
```
### 48 旋转图像
#### 注意按照环来旋转，可以自己画图。 
>另一种方法也可以 首先利用转置矩阵转置 之后每一行反转即可。
```
class Solution {
    public void rotate(int[][] matrix) {
        if(matrix.length==0) return;
        //按照环来进行旋转，从外到内，每次只旋转一个环
        for(int loop=0;loop<matrix.length/2;loop++)//loop代表当前在旋转第几个环，一般我们只需旋转matrix.length/2
        {
        //在每个环中，每次需要旋转四个数的位置，直到这个环的数都被旋转完
            int start = loop;//每个环的开始位置就是环的序号
            int end = matrix.length-start-1;//每个环的结束位置
            for(int i=0;i<end-start;i++)//注意每次旋转环中的所有节点 最后一个不用旋转（最后一个与第一个旋转）
            {//注意逆时针旋转四个元素 会达到最后的效果 注意对每一个环每次是行不变还是列不变
                int temp = matrix[start][start+i];
                matrix[start][start+i] = matrix[end-i][start];
                matrix[end-i][start] = matrix[end][end-i];
                matrix[end][end-i] = matrix[start+i][end];
                matrix[start+i][end] = temp;
            }
        }
        return;
    }
}
```
### 54 螺旋矩阵
#### 此题可以认为是旋转n*n矩阵90度的变形，两者的思路非常一致，都是一环一环输出或者swap,这个不同的是螺旋矩阵不一定是方针 那么循环的次数应该是由row和col中较小的决定 还有要着重考虑每一个环所输出的下标与环的序号之间的关系；特别注意最后两个边的判断情况
```
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int row = matrix.length;
        if(row==0) return res;
          int col = matrix[0].length;
       
        for(int loop=0;loop<(Math.min(row,col)+1)/2;loop++)
        {
        //从左到右
            for(int i=loop;i<=col-1-loop;i++)
                res.add(matrix[loop][i]);
         //从右到下
            for(int j=loop+1;j<row-loop;j++)
                res.add(matrix[j][col-1-loop]);
          //从下到左  注意单行的情况
	    for(int m=col-1-1-loop;m>=loop&&(row-1-loop!=loop);m--)
                res.add(matrix[row-1-loop][m]);
          //从左到上  注意单列的情况
	    for(int n=row-1-1-loop;(n>loop)&&(col-1-loop!=loop);n--)
                res.add(matrix[n][loop]);
        }
        return res;
        }
}
```
### 49 字母异位词分组
#### 注意理解题目的意思 之后我们抽象让所有的异位词对应同一个字典序，制作这样的映射：一个字典序的key对应所有的value(不同顺序） 自然我们想到的hashmap结构应该是一个key 保存字典序 value利用数组存储（List更好操作） 
```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String,List<String>>  hash = new HashMap<>();//hashmap key保存每个字符串的字典序 List<String>存储每个字典序的不同形态
        for(int i=0;i<strs.length;i++)
        {
            char[] str = strs[i].toCharArray();
            Arrays.sort(str);//按字典排序
            String temp = String.valueOf(str);//char转string
            if(!hash.containsKey(temp))
            {
                List<String> res = new ArrayList<>();
                res.add(strs[i]);
                hash.put(temp,res);
            }
            else
            {
                hash.get(temp).add(strs[i]);
            }
        }
        return new ArrayList<List<String>>(hash.values());//此处借用ArrayList的构造方法
    }
}


```
### 50 pow(x,n)
### 暴力法求解无论是递归还是非递归都会有问题  
栈溢出或者超时 所以需要采用的方法为奇偶分离  如果为偶数 xn=x(n/2)*x(n/2)如果为奇数，则为xn=x(n-1)/2 x(n-1)/2 x 
```
class Solution {
    public double myPow(double x, int n) {
        if(n==0) return (double)1;
        if(x==(double)1) return   (double)1;//1的所有次方都是1
        if(x==(double)(-1)&& n<0) return (double)1; //-1的所有负次方都是
        if(n>Integer.MAX_VALUE||n<=Integer.MIN_VALUE) return 0.0;
        boolean flag = true;//flag为正，代表为正数
        double temp = 1;
        if(n<0)
        {
            n = -n;
            flag = false;
        }
                
        while(n>0)
        {
            if(n%2==0)  //如果为偶数
            {
            x *= x;
            n>>=1; //n/2
            }
            else
            {
            temp = temp*x;
            n--;}
            }
        
        return flag==true?temp:1/temp;
        
    }

}

class Solution {
    public double myPow(double x, int n) {
        if(n==0) return (double)1;
        if(x==(double)1) return   (double)1;//1的所有次方都是1
        if(x==(double)(-1)&& n<0) return (double)1; //-1的所有负次方都是
        if(n>Integer.MAX_VALUE||n<=Integer.MIN_VALUE) return 0.0;
        boolean flag = true;//flag为正，代表为正数
        double temp = 1;
        if(n<0)
        {
            n = -n;
            flag = false;
        }                
        temp = help(x,n);
        return flag==true?temp:1/temp;
        
    }
public double help(double x, int n)
{
    if(n==0) return (double)1;
    if(n%2==0) {
    double half = help(x,n/2);
        return half*half;
    }
    else 
    {
        double half = help(x,n/2);
        return half*half*x;
        }
}
}
//help第二种写法  
public double help(double x, int n)
{
    if(n==0) return (double)1;
    if(n%2==0) {
        return help(x*x,n/2);
    }
    else 
    {
        return help(x*x,n/2)*x;
        }
}
```
### 55跳跃游戏
#### way1 动态规划
```

class Solution {
    public boolean canJump(int[] nums) {
        if(nums.length==1) return true; //特殊情况
        boolean[] res = new boolean[nums.length]; //分割点，每一个点是否可以到达
          res[0] = true;
        for(int i=1;i<nums.length;i++)
        {
            for(int j=0;j<i;j++)//遍历i之前的所有的点，只要有一个可以到达则可以到达
            {
                if(res[j]&&j+nums[j]>=i) 
		{
                    res[i] = true;
                    break;
                }
            }
        }
    return res[nums.length-1];
    }
}
```
#### way2 贪心算法
```
class Solution {
    public boolean canJump(int[] nums) {
        int reach = 0; //维护一个可以到达的最远的位置
        for(int i=0;i<nums.length;i++){
        if(i>reach||reach>=nums.length-1) break; //这里是不可达或者已经到达终点的判断条件
            reach = Math.max(reach,i+nums[i]);
        }
    return reach>=nums.length-1;
    }
}
```
### 56 合并区间
### 排序+遍历 
Arrays.sort(intervals, (a, b) -> a[0] - b[0]); //此方法重写sort对比的比较函数
还有new int[0][]的含义
```
class Solution {
    public int[][] merge(int[][] intervals) {
        List<int []> res = new ArrayList<>();
        if(intervals.length==0 || intervals==null)  return res.toArray(new int[0][]); 
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]); //sort是Array的静态方法 将所有按start端点进行排序
        int i=0;
        while(i<intervals.length){
            int start = intervals[i][0];
            int end = intervals[i][1];
            while(i<intervals.length-1 && end>=intervals[i+1][0])//注意防止i越界 第一个i的条件
            {
                    i++;
                    end = Math.max(intervals[i][1],end);
            }
            res.add(new int[]{start,end});//此时i是此次合并的末尾
            i=i+1;//i继续向后合并
        }
        return res.toArray(new int[0][]);
    }
}

```
### Timor Attacking 
### 本质上也是合并区间问题的变形
```
//贪心的思想 计算中毒时间。如果有重合，则只计算一次。贪心算法，比较相邻两个时间点的时间差，如果小于duration，就加上这个差，如果大于或等于，就加上duration即可
class Solution { //9ms
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        if (timeSeries == null || timeSeries.length == 0) return 0;
        int sum = 0;
        int len = timeSeries.length;
        for (int i = 1;i < len;i ++){
            int diff = timeSeries[i] - timeSeries[i - 1];
            sum += (diff < duration) ? diff : duration;
        }
        return sum + duration;//注意这种方法最后一次伤害没有计算进去，需要在加上一个duration 
    }
}


class Solution { //6ms
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        if (timeSeries == null || timeSeries.length == 0 || duration == 0) return 0;
        int result = 0;
        int start = timeSeries[0];
        int end = timeSeries[0] + duration;
        for (int i = 1; i < timeSeries.length; i++) {
            if (timeSeries[i] > end) {
                result += end - start;
                start = timeSeries[i];
            }
            end = timeSeries[i] + duration;
        }
        result += end - start;
        return result;
    }
}
```
### 62 不同路径
#### DP算法
```
class Solution {
    public int uniquePaths(int m, int n) {
        //利用DP算法计算
        int[] [] dp = new int [m] [n];
        for(int i=0;i<m;i++) dp[i][0] = 1;
        for(int i=0;i<n;i++) dp[0][i] = 1;
        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            dp[i][j] = dp[i-1][j]+dp[i][j-1];
        }
    return dp[m-1][n-1];
    }
}
```
### 63 sqrt函数的实现（二分法）
```
class Solution {
    public int mySqrt(int x) {
        if(x==0)  return 0;
        double offset = 0.05d;
        double up = x; //终点
        double low = 0; //起始点
        double key = x; //答案所在位置
        while(true){
            if((key*key+offset)>x && (key*key-offset)<x)
            break;
            else if(key*key>x)
           {up = key;
            key = (up+low)/2;
           }
            else if(key*key<x)
            {low = key;
             key = (up+low)/2;
            }
        }
        return (int) key;
            }
}
```
### 
### 78 子集
##### 先排序 然后没加入一个新的元素，将res中保存的元素全部加上这个新元素  直到最后一个 res中会有所有的元素结果。
```
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> empty = new ArrayList<>();
        res.add(empty);
        Arrays.sort(nums);
        for(int i=0;i<nums.length;i++){
            int size = res.size();
            for(int j=0;j<size;j++)
            {
                List<Integer> temp = new ArrayList<>(res.get(j));//res.get() 获取第j个数组
                temp.add(nums[i]);
                res.add(temp);
            }
        }
        return res; 
    }
}
```
### 79 单词搜索
#### dfs遍历+回溯
```
class Solution {
    public boolean exist(char[][] board, String word) {
        //遍历数组寻找数组起点
        for(int i=0;i<board.length;i++)
        {
            for(int j=0;j<board[0].length;j++)
            {
                if(dfs(i,j,board,word,0)) return true;
            }
        }
        return false;
    }
    //dfs+回溯
    public boolean dfs(int x,int y,char[][] board,String word,int index){
        //判断这个点是否越界
        if(x>=board.length||y>=board[0].length||x<0||y<0) return false;
        //判断这个点是否是word的当前字符
        if(board[x][y]!=word.charAt(index)) return false;
        //如果匹配成功，先看是否是最后一个字符
        if(index==word.length()-1) return true;
        //如果不是这最后一个，则匹配下一个 依次代表上下左右
        //因为不允许重复使用,当前数组位置的字母已经使用过了,则换成其他字母，表示不允许被重复使用
        board[x][y] = '.';
        if(dfs(x-1,y,board,word,index+1)||dfs(x+1,y,board,word,index+1)||dfs(x,y-1,board,word,index+1)||dfs(x,y+1,board,word,index+1))  return true;
        //之后记得复原
        board[x][y] = word.charAt(index);
        return false;//如果到这里还没有返回true 则说明不对
    }
}
```
### 91 解码方法
#### DP算法
```
class Solution {
    public int numDecodings(String s) {
        if(s.charAt(0)=='0') return 0;//处理特殊情况,第一个若为0则返回 不用检测
        int len = s.length();
        int[] dp = new int[len+1];
        dp[0] = 1;
        dp[1] = 1;
        
        for(int i=2;i<=s.length();i++)
        {
            if(s.charAt(i-1)!='0')
            dp[i] += dp[i-1];
            if(s.charAt(i-2)=='1'||(s.charAt(i-2)=='2' && s.charAt(i-1)<'7' ))
            dp[i] += dp[i-2];
        }
        return dp[len];
    }
}
```
### 94 二叉树中序遍历
#### 递归解法  https://leetcode-cn.com/problems/binary-tree-inorder-traversal/solution/die-dai-fa-by-jason-2/
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
       List<Integer> res = new ArrayList<>();
       help(root,res);
       return res;
    }
    public void help(TreeNode root,List<Integer> res){
        if(root==null) return;
        if(root.left!=null) 
        help(root.left,res);
        res.add(root.val);
        if(root.right!=null) 
        help(root.right,res);
    }
}
```
#### 迭代解法
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
       List<Integer> res = new ArrayList<>();
       Stack<TreeNode> s = new Stack<>();
      while(root!=null||!s.empty()){
      while(root!=null){
           s.push(root);
           root = root.left;
       }
       TreeNode t = s.pop();
        res.add(t.val);
        root = t.right;
      }
       return res; 
    }
}
```
### 144 前序遍历
#### 递归
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        help(root,res);
        return res;
    }
    public void help(TreeNode root,List<Integer> res){
        if(root==null) return;
        res.add(root.val);
        if(root.left!=null)
        help(root.left,res);
        if(root.right!=null)
        help(root.right,res);
    }
}
```
#### 迭代
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> s = new Stack<>();
        while(root!=null||!s.empty())
        {
            while(root!=null)
            {
                res.add(root.val);
                s.push(root);
                root=root.left;
            }
            TreeNode temp = s.pop();
            root = temp.right;
        }
        return res;
    }}
 ```
 ### 145 后序二叉树
 #### 递归
 ```
 /**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        help(res,root);
        return res;
    }
    public void help( List<Integer> res,TreeNode root){
        if(root==null) return;
        if(root.left!=null)
        help(res,root.left);
        if(root.right!=null)
        help(res,root.right);
        res.add(root.val);
    }
}
 ```
 #### 迭代
 ```
 /**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> s = new Stack<>();
        while(!s.empty()||root!=null)
        {
        while(root!=null)
        {
            res.add(root.val);
            s.push(root);
            root = root.right;
        }
        TreeNode temp = s.pop();
        root = temp.left;
        }
        Collections.reverse(res);//前序 根左右  后序 左右根  所以借鉴前序的遍历方法  转为根右左 最后反转即可
        return res;
}}
 ```
 ### 98 是否为搜索二叉树
 #### 搜索二叉树的中序遍历一定是升序的
 ```
 /**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
      long num = Long.MIN_VALUE;
      boolean flag = true;
    public boolean isValidBST(TreeNode root) {
      
        if(root==null) return true;
        if(root.left!=null)
        isValidBST(root.left);
        if(num>=root.val) flag = false;
        else {
            num = root.val;
        }
        if(root.right!=null)
        isValidBST(root.right);
        return flag;
    }
}
 ```
#### 迭代版
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
      long num = Long.MIN_VALUE;
      boolean flag = true;
      Stack<TreeNode> s = new Stack<>();
    public boolean isValidBST(TreeNode root) {
      
        while(!s.empty()||root!=null){
            while(root!=null)
            {
                s.push(root);
                root= root.left;
            }
            TreeNode temp = s.pop();
            if(num>=temp.val) flag = false;
            else num = temp.val;
            root = temp.right;
        }
        return flag;
    }
}
```
### 二叉树层次遍历
### 注意每一层都需要放在单独的数组内，所以要用一个level来记录目前在哪一层
```2
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {

    public List<List<Integer>> levelOrder(TreeNode root) {
            List<List<Integer>> res = new ArrayList<List<Integer>>();
            help(res,root,0);
            return res;
    }
    public void help(List<List<Integer>> res,TreeNode root,int level){
        if(root==null) return;
        if(level>=res.size())
        {
            res.add(new ArrayList<Integer>());
        }
        res.get(level).add(root.val);
        help(res,root.left,level+1);
        help(res,root.right,level+1);
    }
}

```
### 锯齿状遍历二叉树
#### 借鉴层次遍历结果 将偶数层反转即可
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
            help(res,root,0);
            for(int i=2;i<=res.size();i+=2)
            {
                Collections.reverse(res.get(i-1));
            }
            return res;
    }
    public void help(List<List<Integer>> res,TreeNode root,int level){
        if(root==null) return;
        if(level>=res.size())
        {
            res.add(new ArrayList<Integer>());
        }
        res.get(level).add(root.val);
        help(res,root.left,level+1);
        help(res,root.right,level+1);
    }
    
}
```
### 重建二叉树
#### 难
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
    class Solution {
  
        public TreeNode buildTree(int[] preorder, int[] inorder) {
            if(preorder == null || inorder == null || preorder.length==0){
                return null;
            }
        Map<Integer,Integer> res = new HashMap<>();
        if(preorder.length==0) return null;
        for(int i=0;i<inorder.length;i++)
        {
            res.put(inorder[i],i);
        }

            return buildCore(preorder,0,preorder.length-1,inorder,0,inorder.length-1,res);
        }
        private TreeNode buildCore(int[] preorder,int preSt,int preEnd,int[] inorder,int inSt,int inEnd, Map<Integer,Integer> res){
            //前序遍历第一个节点是根节点
            int rootValue = preorder[preSt];
            TreeNode root = new TreeNode(rootValue);

            //前序序列只有根节点
            if(preSt == preEnd){
                return root;
            }
            //遍历中序序列，找到根节点的位置
            int rootInorder = res.get(rootValue);
            //左子树的长度
            int leftLength = rootInorder - inSt;
            //前序序列中左子树的最后一个节点
            int leftPreEnd = preSt + leftLength;

            //左子树非空
            if(leftLength>0){
                //重建左子树
                root.left = buildCore(preorder,preSt+1,leftPreEnd,inorder,inSt,inEnd,res);
            }
            //右子树非空
            if(leftLength < preEnd - preSt){
                //重建右子树
                root.right = buildCore(preorder,leftPreEnd +1,preEnd,inorder,rootInorder+1,inEnd,res);
            }
            return root;
        }
    }


/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer,Integer> res = new HashMap<>();
        if(preorder == null || inorder == null || preorder.length==0
) return null;
        for(int i=0;i<inorder.length;i++)
        {
            res.put(inorder[i],i);
        }
        return help(preorder,0,preorder.length-1,inorder,0,inorder.length-1,res);
    }
    public TreeNode help(int[] preorder, int pre_start, int pre_end, int[] inorder, int ino_start, int ino_end, Map<Integer,Integer> res)
    {
        TreeNode root = new TreeNode(preorder[pre_start]);
        if(pre_start==pre_end) return root;
        int location = res.get(preorder[pre_start]);
        int length = location-ino_start;
       if(length>0)
    root.left =  help(preorder,pre_start+1,pre_start+length,inorder,ino_start,ino_end,res);
       if(length<pre_end-pre_start)
    root.right = help(preorder,pre_start+length+1, pre_end,inorder,location+1,ino_end,res);
        return root;
    }
}

```
### 被围绕的区域
#### DFS遍历  先从边界处理 找到不用修改的o点  利用dfs找到关联的  最后遍历替换就可
```
class Solution {
    public void solve(char[][] board) {
        if(board == null|| board.length == 0 || board[0].length == 0 )
        return;
        //遍历第一行和最后一行
        for(int i=0;i<board[0].length;i++)
        {
            if(board[0][i]=='O'){
                search(board,0,i);
            }
            if(board[board.length-1][i]=='O'){
                search(board,board.length-1,i);
            }
            
        }
        //遍历第一列和最后一列
        for(int i=0;i<board.length;i++)
        {
            if(board[i][0]=='O'){
                search(board,i,0);
            }
            if(board[i][board[0].length-1]=='O'){
                search(board,i,board[0].length-1);
            }
        }
        //替换
        for(int i=0;i<board.length;i++)
        {
            for(int j=0;j<board[0].length;j++)
            {
                if(board[i][j]=='*')
                board[i][j] = 'O';
               else if(board[i][j]=='O')//这里不能用if 否则上面的if将其修改为O后 又会被下面的修改为X
                board[i][j] = 'X';
            }
        }
    }
    public static void search(char[][] board,int row,int col){
        if(row<0 || col<0 || row>=board.length || col>=board[0].length)
        return;
        //如果此处没有O 递归退出
        if(board[row][col]!='O')
        return;
        board[row][col] = '*';
        search(board,row-1,col);
        search(board,row+1,col);
        search(board,row,col-1);
        search(board,row,col+1);
    }
}
```
### 200 岛屿数量
#### 与上面的题非常类似  同样的思路可以借鉴
```
class Solution {
    public int numIslands(char[][] grid) {
        //DFS遍历 1无非两种  第一个找到的1是一个岛 然后利用BFS上下左右如果是1则改为0 如果不是则返回
        if(grid==null||grid.length==0)
        return 0;
        int row = grid.length;
        int col = grid[0].length;
        int result = 0;
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                if(grid[i][j]=='1')
                  {
                      result +=1;
                      bfs(i,j,row,col,grid);
                  }
            }
        }
        return result;
    }
    public void bfs(int i,int j, int row, int col, char[][] grid)
    {
        if(i<0||i>=row||j<0||j>=col)
        return;
        if(grid[i][j]=='0')
        return;
        grid[i][j] = '0';
        bfs(i-1,j,row,col,grid);
        bfs(i+1,j,row,col,grid);
        bfs(i,j-1,row,col,grid);
        bfs(i,j+1,row,col,grid);
    }
}
```
### 分割回文串
#### 采用DFS和回溯法 
```
class Solution {
    List<List<String>> result = new ArrayList<>();
    public List<List<String>> partition(String s) {
        back(s,0,new ArrayList<>());
        return result;
    }
    void back(String s,int index,List<String> list)
    {
        //递归出口
        if(index == s.length())
        {
            result.add(new ArrayList<>(list));
            return;
        }
        for(int i=index;i<s.length();i++)
        {
            String sub = s.substring(index,i+1);
            if(huiwen(sub)){
                list.add(sub);
                back(s,i+1,list);//如果此sub是回文串（从index到i），则遍历此时i+1后面的字符串看是否是回文串
                list.remove(list.size()-1);
            }
        }
    }
    boolean huiwen(String str){
        for(int i=0;i<=str.length()/2;i++)
        {
            if(str.charAt(i)!=str.charAt(str.length()-1-i))
            return false;
        }
        return true;
    }
}
```
### 134 加油站
#### 只需要遍历一次 
a. 最开始，站点0是始发站，假设车开出站点p后，油箱空了，假设sum1 = diff[0] +diff[1] + … + diff[p]，可知sum1 < 0；
　　b. 根据上面的论述，我们将p+1作为始发站，开出q站后，油箱又空了， 设sum2 = diff[p+1] +diff[p+2] + … + diff[q]，可知sum2 < 0。
　　c. 将q+1作为始发站，假设一直开到了未循环的最末站，油箱没见底儿，设sum3 = diff[q+1] +diff[q+2] + … + diff[size-1]，可知sum3 >= 0。
　　要想知道车能否开回 q 站，其实就是在sum3 的基础上，依次加上 diff[0] 到 diff[q]，看看sum3在这个过程中是否会小于0。但是我们之前已经知道 diff[0] 到 diff[p-1] 这段路，油箱能一直保持非负，因此我们只要算算sum3 + sum1是否 <0，就知道能不能开到 p+1站了。
　　如果能从p+1站开出，只要算算sum3 + sum1 + sum2 是否 < 0，就知都能不能开回q站了。
　　因为 sum1, sum2 都 < 0，因此如果 sum3 + sum1 + sum2 >=0 那么sum3 + sum1 必然 >= 0，也就是说，只要sum3 + sum1 + sum2 >=0，车必然能开回q站。而sum3 + sum1 + sum2 其实就是 diff数组的总和 Total，遍历完所有元素已经算出来了。
　　因此 Total 能否 >= 0，就是是否存在这样的站点的 充分必要条件。
　　这样时间复杂度进一步降到了 O(n)。
```
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if(gas.length!=cost.length || gas==null || cost==null)
        return -1;
        int count = 0; //油箱里面剩余的油
        int sum = 0; //从start位置开始，加的油和消耗的油的总差值
        int start = 0;
        for(int i=0;i<gas.length;i++)
        {
            count += gas[i]-cost[i];
            if(sum<0)
            {
                sum = gas[i]-cost[i];
                start = i;
            }
            else
            sum += gas[i]-cost[i]; 
        }
        return count>=0?start:-1;
    }
}
```
### 136 只出现一次的数字
#### hashmap 
```
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<nums.length;i++)
        {
            if(map.containsKey(nums[i]))
            {
                int val = map.get(nums[i]);
                val++;
                map.put(nums[i],val);
            }
            else
            {
                map.put(nums[i],1);
            }
        }
        for(int i=0;i<nums.length;i++)
        {
            if(map.get(nums[i])==1)
             res =  nums[i];
        }
         return res;
        }
    }
```
#### 异或运算 充分考虑题目中其他数字都出现两次
```
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        for(int i=0;i<nums.length;i++)
        res ^=nums[i];
            return res;
    }
    }
```
### 146 LRU缓存机制
#### 利用双向链表（空头、空尾）实现LRU，每次使用一次就删除节点，然后把此节点放在链表头部，之后利用hashmap存储节点降低复杂度
```
class LRUCache {
    //双向链表
    private class Node{
        private int key;
        private int value;
        private Node left;
        private Node right;
        public Node(){};
        public Node(int key,int value)
        {
            this.key = key;
            this.value = value;
        }
    }
    //以下定义是LRUCache类中的
    private Node FirstTemp = new Node();
    private Node LastTemp = new Node();
    Map<Integer,Node> map = new HashMap<>();
    private int capacity;
    private int size;
    public LRUCache(int capacity) //此类的构造函数
    {
        FirstTemp.right = LastTemp;
        LastTemp.left = FirstTemp;
        this.capacity = capacity;
        size = 0;
    }
    public void del(Node node)
    {
     Node lefttmep = node.left;
     Node righttemp = node.right;
     lefttmep.right = righttemp;
     righttemp.left = lefttmep;
     node.left = null;
     node.right = null;
    }
    public void add(Node node)
    {
        Node temp = FirstTemp.right;
        node.left = FirstTemp;
        node.right = temp;
        FirstTemp.right = node;
        temp.left = node;
    }
    public int get(int key) //每次使用时先删除这个节点 然后加到最前面 保证每次删除最后一个节点是最近最少使用的页面
    {
        Node node = map.get(key);
        if(node==null)
        return -1; 
        del(node); 
        add(node);
        return node.value;
    }
    public void put(int key, int value) {
        Node node = map.get(key);
        //如果map中有这个key 则更新key对应的value
        if(node!=null)
        {
            node.value = value;
            del(node);
            add(node);
        }
        else{
            if(size<capacity){
                size++;
            }
            else
            {
                Node tnode = LastTemp.left;
                map.remove(tnode.key);//删除最后一个节点 在map中
                del(tnode);//删除链表节点
            }
            Node newnode = new Node(key,value);
            add(newnode);
            map.put(key,newnode); 
        }
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```
### 148 排序链表 
#### 归并 
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        if(head==null){
            return  null;
        }
        if(head.next==null)
        return  head;
        //利用快慢指针寻找中间节点
        ListNode slow = head;
        ListNode fast = head;
        while(fast.next!=null)
        {
            fast = fast.next;
            if(fast.next==null){
                break;
            }
            else{
                fast = fast.next;
                slow = slow.next;
            }
        }
        //找到左右节点之后分割链表
        ListNode l1 = head;
        ListNode l2 = slow.next;
        slow.next = null; //l1结尾为slow
        //merge
        l1 = sortList(l1);
        l2 = sortList(l2);
        return merge(l1,l2);
    }
    public ListNode merge(ListNode leftlist,ListNode rightlist)
    {
        if(leftlist==null)
        return rightlist;
        if(rightlist==null)
        return leftlist;
        //确定链表头
        ListNode h1 = leftlist;
        ListNode h2 = rightlist;
        ListNode res = null; //此点只用于返回  
        if(h1.val<h2.val)
        {res = h1;
        h1 = h1.next;
        }
        else
        {
            res = h2;
            h2 = h2.next;
        }
        ListNode p = res;
        while(h1!=null && h2!=null)
        {
            if(h1.val<h2.val)
            {
                p.next = h1;
                h1 = h1.next;
                p = p.next;
            }
            else{
                p.next = h2;
                h2 = h2.next;
                p = p.next;
            }
        }
        if(h1!=null)
        p.next = h1;
        if(h2!=null)
        p.next = h2;
    return res;
    }
}
```
### 150 逆波兰表达式
#### 利用辅助stack
```
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> s = new Stack<>();
        for(String string:tokens)
        {
            if(string.equals("+"))
            {
                s.push(s.pop()+s.pop());
            }
            else if(string.equals("-"))
            {
                s.push(-s.pop()+s.pop());
            }
            else if(string.equals("*"))
            {
                s.push(s.pop()*s.pop());
            }
            else if (string.equals("/"))
            {   int temp = s.pop();
                s.push(s.pop()/temp);
            }
            else
            s.push(Integer.parseInt(string));
        }
        return s.pop();

    }
}
```
### 152 乘积最大子序列
#### DP 注意考虑负数 则需要引入dpmin用来保存当nums[i]为负数并且dpmax<0时最大值应该反转为dpmin[i-1]*nums[i] 之后不需要if else 因为核心目的是找到最大的 所以利用Math.max寻找即可。 dpmin的更新可以参考dpmax
```
如果nums[i]大于等于0
dpmax[i-1]>0
dpmax[i] = dpmax[i-1]*nums[i]
dpmax[i-1] <0
dpmax[i] = nums[i]
如果nums[i]<0
dpmax[i-1]>0
dpmax[i] = nums[i]
dpmax[i-1]<0
dpmax[i] =dpmin[i-1]*nums[i]
```
```
class Solution {
    public int maxProduct(int[] nums) {
        if(nums.length==0) return 0;
        int MinResult = nums[0];
        int MaxResult = nums[0];
        int result = nums[0];
        for(int i=1;i<nums.length;i++)
        {
            int temp = MaxResult;//保存dpmax[i-1]用于更新MinResult
            MaxResult = Math.max(MaxResult*nums[i],Math.max(nums[i],MinResult*nums[i]));
            MinResult = Math.min(temp*nums[i],Math.min(nums[i],MinResult*nums[i]));
            if(MaxResult>result)
            result = MaxResult;
        }
        return result;
    }
}
```
### 162寻找峰值
#### 只需遍历一次即可
```
class Solution {
    public int findPeakElement(int[] nums) {
        for(int i=0;i<nums.length-1;i++)
        {
            //只需要看此点是否大于下一点，如果不满足则说明此点小于下一点，说明是一个上坡的趋势，一旦出现此点大于下一点则一定是拐点。如果一直没有则是一个增函数，返回最后一点即可。
            if(nums[i]>nums[i+1])
            return i;
        }
        return nums.length-1;
    }
}
```
### 171 EXCEL转换
```
class Solution {
    public int titleToNumber(String s) {
       //类比十进制的算法  325=10(3*10+2)+5 迭代进行
       int res = 0;
        for(int i=0;i<s.length();i++)
        {
        res = res*26+s.charAt(i)-'A'+1;
        }
        return res;
    }
}
```
```
普通的计算方法
class Solution {
    public int titleToNumber(String s) {
       int res = 0;
        for(int i=0;i<s.length();i++)
        {
        res += (s.charAt(i)-'A'+1)*Math.pow(26,s.length()-1-i);
        }
        return res;
    }
}
```
### 166 分数到小数
#### 模拟除法的整个过程，注意循环，注意溢出，注意符号
```
https://leetcode-cn.com/problems/fraction-to-recurring-decimal/solution/fen-shu-dao-xiao-shu-by-leetcode/
class Solution {
    public String fractionToDecimal(int numerator, int denominator) {
        if(numerator==0) return new String("0");
        //此题目不考虑抛异常 也就是除数为0的情况
        StringBuilder res = new StringBuilder();
        //利用异或判断结果符号 
        if(numerator<0^denominator<0)
            res.append("-");
        //转为long避免溢出 因为int范围为 -2^31~~~2^31 -1
        long  dividend = Math.abs(Long.valueOf(numerator));
        long  divisor = Math.abs(Long.valueOf(denominator));
        res.append(String.valueOf(dividend/divisor));//将整数部分输入
        long  remain = dividend%divisor; //获得余数
        if(remain==0)//如果余数为0则可以整除
        {return res.toString();}
        res.append(".");
    Map<Long,Integer> map = new HashMap<>();//利用hashmap的key存储小数部分的内容，value存储位置用于插入（）
        while(remain!=0)
        {
         if(map.containsKey(remain))
         {
             res.insert(map.get(remain),"(");
             res.append(")");
             break;
         }
         //如果map中没有，则说明没有循环
         map.put(remain,res.length());//length这个点就是之后要插入的数组位置
         remain *=10;
         res.append(String.valueOf(remain/divisor));
         remain = remain%divisor;
        }
        return res.toString();
    }
}
```
### 172 阶乘后的0
```
class Solution {
    public int trailingZeroes(int n) {
    //一个数因数分解寻找其后的0只和2,5的个数有关。2肯定多于5 所以只要找5的个数就行 n/5+n/25+n/125+...
     int cnt = 0;
     while(n!=0)
     {
         cnt += n/5;
         n=n/5;
     }
     return cnt;
}}
```
### 179 最大数
#### 主要是自定义排序规则
```
class Solution {
    public  class Compare implements Comparator<String>{
        public int compare(String a,String b)
        {
            return (b+a).compareTo(a+b);//如果b+a>a+b 返回true 则b比a大  如a=10 b=2 排序结果应为ba 而不是ab (普通排序结果为ab)
        }
    }
    public String largestNumber(int[] nums) {
        //将nums转为String
        String[] num = new String[nums.length];
        for(int i=0;i<nums.length;i++)
        {
            num[i] = String.valueOf(nums[i]);
        }
        //利用自定义规则进行排序
        Arrays.sort(num,new Compare());
        if(num[0].equals("0"))
        return new String("0");
            StringBuilder res = new StringBuilder();
            for(String temp:num){
                res.append(temp);
            }
            return res.toString();
    }
}
```
### 190颠倒二进制位
```
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int value = 0;
        //思路：做一个32次循环，将数翻转即可 注意利用二进制运算解决
        for(int i=0;i<32;i++)
        {
            if((n&1)==1)//n的低位与1做与运算 
           {
               value = (value<<1) + 1;//左移并加1（因为此位是1）  注意优先级 加括号
               n=n>>1;
           } 
            else  //此位是0
            {
                value = value<<1; //左移一位即可
                n=n>>1;
            }
        }
        return value;
    }
}
```
### 191 1的个数
### 与190类似
```
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int cnt = 0;
        for(int i=0;i<32;i++)
        {
            if((n&1)==1)
            {
                cnt++;
                n = n>>1;
            }
            else
             n = n>>1;
        }
        return cnt;
    }
}
```
### 198 打家劫舍
#### dp算法
```
class Solution {
    public int rob(int[] nums) {
        int[] dp = new int[nums.length+2];
        dp[0] = 0;
        dp[1] = 0;
        for(int i=0;i<nums.length;i++)
        {
            dp[i+2] = Math.max(dp[i]+nums[i],dp[i+1]);
        }
        return dp[nums.length+2-1];
    }
}
```
### 202 快乐数
```
class Solution {
    public boolean isHappy(int n) {
        Set<Integer> temp = new HashSet<>();
        int value = 0;
        while(true){
            while(n!=0){
               value += Math.pow(n%10,2);
               n = n/10;
            }
            if(value==1)
            return true;
            if(temp.contains(value))
            return false;
            //没有重复的 则还可以计算 
            temp.add(value);
            n = value;
            value = 0;
        }
    }
}
```
### 207 课程表
```
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //此题本质上是有向无环图DAG的判断问题 若此图有环则不符合条件
        int [] degree = new int[numCourses];//制作一个degree表存取每个课程的入度  二维数组中每一个都是一个边 从1指向0  每个1下标对会给0下标增加一个入度
        for(int [] arr: prerequisites)
        {
            degree[arr[0]]++;//每个1下标对会给0下标增加一个入度，degree对应arr[0]的课程节点度+1
        }
        LinkedList<Integer> queue = new LinkedList<>();
        for(int i=0;i<degree.length;i++)
        {
            if(degree[i]==0) queue.add(i); //degree数组的下标为课程的编号,将入度为0的课程存储在queue中
        }
        while(!queue.isEmpty()){
            Integer pre =  queue.removeFirst();//出队列中第一个
            numCourses--;//每次出队列一个说明安排完一门课程
            for(int [] req:prerequisites) //遍历prerequisties 寻找此课程所贡献的入度 根据拓扑排序规则 此课程贡献的入度全部删除 也就是其所有链接的点的入度减一 如果减完入度为0则将此课程加入queue 否则继续 如果能安排完所有的课程 则说明成功  
            {
                if(req[1]!=pre) continue;
                else
                {
                    degree[req[0]]--;
                    if(degree[req[0]]==0)
                    queue.add(req[0]);
                }
            }
        }
        if(numCourses==0)
        return true;
        else
        return false;
    }
}

```
###210 课程表2  与1不同的是需要输出课表排序列表
```
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
                //此题本质上是有向无环图DAG的判断问题 若此图有环则不符合条件
        int [] degree = new int[numCourses];//制作一个degree表存取每个课程的入度  二维数组中每一个都是一个边 从1指向0  每个1下标对会给0下标增加一个入度
         //ArrayList<Integer> result = new ArrayList<>();
         int[] result = new int[numCourses];
         int index = 0;
        for(int [] arr: prerequisites)
        {
            degree[arr[0]]++;//每个1下标对会给0下标增加一个入度，degree对应arr[0]的课程节点度+1
        }
        LinkedList<Integer> queue = new LinkedList<>();
        for(int i=0;i<degree.length;i++)
        {
            if(degree[i]==0) queue.add(i); //degree数组的下标为课程的编号,将入度为0的课程存储在queue中
        }
        while(!queue.isEmpty()){
            Integer pre =  queue.removeFirst();//出队列中第一个
            numCourses--;//每次出队列一个说明安排完一门课程
            result[index++] = pre;
            for(int [] req:prerequisites) //遍历prerequisties 寻找此课程所贡献的入度 根据拓扑排序规则 此课程贡献的入度全部删除 也就是其所有链接的点的入度减一 如果减完入度为0则将此课程加入queue 否则继续 如果能安排完所有的课程 则说明成功  
            {
                if(req[1]!=pre) continue;
                else
                {
                    degree[req[0]]--;
                    if(degree[req[0]]==0)
                    queue.add(req[0]);
                }
            }
        }
        if(numCourses==0)
        {
        return result;
        }
        
        else
        return new int[0];
    }
}
```
### 204 计数质数 （小于n的质数，不包括n）
```
class Solution {
    public int countPrimes(int n) {
        boolean [] film = new boolean[n];
        int result = 0;
        //一个数是质数，则它的倍数都不可能是质数
       Arrays.fill(film,true);
        for(int i=2;i<n;i++){
            if(film[i]==true)
            {
                for(int j=i*2;j<n;j+=i)
                film[j] = false;
            }
        }
       for(int i=2;i<n;i++)
       {
           if(film[i]==true)
           result++;
       }
        return result;
    }
}
```
### 208 trie树
#### 理解字典树、前缀树的组织方式 尤其是树的数据结构 与二叉树不同的是 他需要一个26的一维数组 
```
class Trie {
    class TrieNode{
        private boolean status;//指定此点对应的单词是否在前缀树中存在 
        private TrieNode[] nexts; 
        public TrieNode(){
            status = false;
            nexts = new TrieNode[26];
        }
    }
    //root节点
    TrieNode root = new TrieNode();
    //此树以root为起点，沿途的树组织起来，在最后一个节点利用status表明此单词存在 沿途其他的不关注 也不赋值 本质使用单词的字典序定位
    /** Initialize your data structure here. */
    public Trie() {
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        if(word==null)
        return;
        TrieNode node = root;
        for(int i=0;i<word.length();i++){
            int index = word.charAt(i)-'a';
            if(node.nexts[index] == null)
                node.nexts[index] = new TrieNode();
            node = node.nexts[index];
        }
        node.status = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        if(word==null)
        return false;
           TrieNode node = root;
        for(int i=0;i<word.length();i++){
            int index = word.charAt(i)-'a';
            if(node.nexts[index]==null)
              {return false;}  
            node = node.nexts[index];
        }
        return node.status;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
      if(prefix==null)
        return false;
           TrieNode node = root;
        for(int i=0;i<prefix.length();i++){
            int index = prefix.charAt(i)-'a';
            if(node.nexts[index] == null)
                return false;
            node = node.nexts[index];
        }
        return true;   
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```
### 212 单次搜索 DFS+Trie 
```
class Solution {
    List<String> result = new ArrayList<>();
    public List<String> findWords(char[][] board, String[] words) {
        TrieNode trieNode = buildTrie(words);
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs (board, i, j, trieNode);
            }
        }
        return result;
    }
    public void dfs(char[][] board, int i, int j, TrieNode trieNode) {
        char c = board[i][j];
        if ('#' == c || null == trieNode.next[c - 'a']) return;
        trieNode = trieNode.next[c - 'a'];
        if (null != trieNode.word) {
            result.add(trieNode.word);
            trieNode.word = null;     //去重
        }
        board[i][j] = '#';
        if (i > 0) dfs(board, i - 1, j ,trieNode);
        if (j > 0) dfs(board, i, j - 1, trieNode);
        if (i < board.length - 1) dfs(board, i + 1, j, trieNode);
        if (j < board[0].length - 1) dfs(board, i, j + 1, trieNode);
        board[i][j] = c;
    }
    private class TrieNode {
        private TrieNode[] next = new TrieNode[26];
        private String word;
    }
    public TrieNode buildTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode trieNode = root;
            for (char c : word.toCharArray()) {
                int i = c - 'a';
                if (null == trieNode.next[i]) {
                    trieNode.next[i] = new TrieNode();
                }
                trieNode = trieNode.next[i];
            }
            trieNode.word = word;
        }
        return root;
    }
}

```
### 215 数组中第K大的元素
#### 优先队列是一种数据结构 底层依托堆排序 因为PriorityQueue默认升序 则数组最后一位是最大的 保持头部一直是第K大元素
```
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        //优先队列默认升序 
        for(int val:nums){
            pq.add(val);
            if(pq.size()>k)
            pq.poll();
        }
        return pq.poll();
    }
}
```
### 378 有序矩阵中第k小的元素
#### 注意优先队列的compare方法 
```
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                return b-a; 
                //注意: a-b 是升序排列  b-a是降序排列      
                }
        });
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                q.add(matrix[i][j]);
                if (q.size() > k) q.poll();
            }
        }
        return q.poll();
    }
}
```
###  347 数组中前k大元素
```
class Solution {
public List<Integer> topKFrequent(int[] nums, int k) {
        // 使用字典，统计每个元素出现的次数，元素为键，元素出现的次数为值
        HashMap<Integer,Integer> map = new HashMap();
        for(int num : nums){
            if (map.containsKey(num)) {
               map.put(num, map.get(num) + 1);
             } else {
                map.put(num, 1);
             }
        }
        // 遍历map，用最小堆保存频率最大的k个元素
        PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                return map.get(a) - map.get(b); 
                //重写的原因在于要对比的是a和b在map中对应的value  
                //注意: a-b 是升序排列  b-a是降序排列 
                //此处需要与最小的对比 则需要升序排列 但不是对比ab本身         
                }
        });
        for (Integer key : map.keySet()) {
            if (pq.size() < k) {
                pq.add(key);
            } else if (map.get(key) > map.get(pq.peek())) {
                pq.remove();
                pq.add(key);
            }
        }
        // 取出最小堆中的元素，出现的频次从小到大
        int [] temp = new int[k];
        for(int i=0;i<k;i++)
        {
            temp[i] = pq.remove();
        }
        List<Integer> res = new ArrayList<>();
        for(int u=k-1;u>=0;u--)
        {
            res.add(temp[u]);
        }
        return res;
    }
}


```

### 230 二叉搜索树中第K小元素
#### 中序遍历是升序的结果排序 
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int num,res;
    public int kthSmallest(TreeNode root, int k) {
            num = k;
            search(root);
            return res;
    }
    public void search(TreeNode root)
    {
            if(root==null || num<0) return;
            search(root.left);
            num--;
            if(num==0)  
            res = root.val;
            search(root.right);
    }
}
```
### 203 计算器II
#### 去除空格 先算乘除  再算加减
```
class Solution {
    public int calculate(String s) {
        s = s.replaceAll(" ","");
        int len = s.length();
        int num = 0;
        int  result =0;
        char flag = '+';
        Stack<Integer> stack = new Stack<>(); 
        for(int i=0;i<s.length();i++)
        {
            if(s.charAt(i)>='0'){
                num = num*10 + (s.charAt(i)-'0');
            }
            if(s.charAt(i)<'0'||i ==s.length()-1)
            {
                if(flag=='+')     stack.push(num);
                else if(flag=='-')     stack.push(-num);
                else
                {
                    int temp = (flag=='*')?stack.peek() * num: stack.peek()/num;
                    stack.pop();
                    stack.push(temp);
                }
                //更新num与flag
                num = 0;
                flag = s.charAt(i);
            }
        }
        while(!stack.empty())
        {
            result += stack.pop();
        }
        return result;
    }
}
```
### 二叉树的最近公共祖先（可以为两个节点中的一个）
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
       //考虑递归出口 若root为空则返回null 若root为p或者q则为结果  
       //pq不为空也不是root节点则遍历左子树与右子树 重复此过程 会有四种情况 
       if(root==null||root==p||root==q) return root; 
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left==null&&right==null) return null; //都找不到 则返回null
        if(left==null&&right!=null) return right; //right有p或者q 则返回right
        if(left!=null&&right==null) return left;// left有p或者q 则返回left
        return root; //如果都有 则返回root 
    }
}
```
### 238 除自身以外的数组乘积 (不可用除法）
```
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int[] left = new int[nums.length];
         int[] right = new int[nums.length];
         int [] result = new int[nums.length]; 
         left[0] = nums[0];
         right[nums.length-1] = nums[nums.length-1];
         for(int i=1;i<nums.length;i++)
         {
             left[i] = left[i-1]*nums[i];
         }
        for(int i=nums.length-1-1;i>=0;i--)
         {
             right[i] = right[i+1]*nums[i];
         }
         for(int i=0;i<nums.length;i++){
             if(i==0) result[i] = right[i+1];
             else if(i==nums.length-1) result[i] = left[i-1];
             else
             result[i] = left[i-1]* right[i+1];
         }
         return result;
    }
}
```
### 搜索二维数组
```
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
    //注意从右上角开始搜索
        if(matrix==null||matrix.length<1||matrix[0].length<1) return false; 
        int row = 0, col = matrix[0].length - 1;
        //row col合格
         while(row>=0&&row<matrix.length&&col>=0&&col<matrix[0].length)
        {
            if(matrix[row][col]>target)
              col--;
            else if(matrix[row][col]<target)
              row++;
            else 
                return true;
        }
        return false;
    }
}
```

### 242 有效的字母异位词
#### 保证每个符号出现次数一直
```
class Solution {
    public boolean isAnagram(String s, String t) {
        if(s.length()!=t.length())
        return false;
        int[] count = new int[26];
        for(int i=0;i<s.length();i++)
        {
            count[s.charAt(i)-'a']++;//s对应的id++
            count[t.charAt(i)-'a']--;//t对应的id--，最后count中都是0就是相同的
        }
        for(int c:count)
        {
            if(c!=0)
            return false;
        }
        return true;
    }
}
```
### 279 完全平方数
#### dp没看懂  用的数学方法 任何一个数最多只需要四个完全平方数来表示
```
class Solution {
    public int numSquares(int n) {
        while(n%4==0)
        {
            n /= 4;}
            if(n%8==7)
            return 4;
            for(int i=1;i<=n;i++)
            {
                if(i*i==n)
                return  1;
            }
            for(int i=1;i*i<=n;i++)
            {
                int temp = (int)Math.sqrt(n-i*i);
                if((i*i+temp*temp)==n && i>0&&temp>0)
                return 2;
            }
            return 3;
        }
    }
```
### 寻找重复数 不可修改元素组 空间复杂度为1 
#### 巧妙转换为二分查找问题  简单思路 为先排序 然后遍历一遍  如果那个数跟前面重复了则说明找到了
```
class Solution {
    public int findDuplicate(int[] nums) {
        int left = 1;
        int right = nums.length-1;
        int mid = 0;
        int count = 0;
        while(left<right){
            mid = left+ (right-left)/2;
            count = 0;
            for(int num:nums)
            {
                if(num<=mid)
                count++;
            }
            if(count<=mid)
            {
                left = mid+1;
            }
            if(count>mid)
            {
                right = mid;
            }
        }
        return left;
    }
}
```
### 289 生命游戏
### 因为不可以用覆盖此个点之前的状态 因为还要用于后续计算 所以引入二位数编码  后续解码即可 主要就是更新细胞的状态
```
class Solution {
    public void gameOfLife(int[][] board) {
        int row = board.length;
        int col = board[0].length;
        //00（代表修改前为0，修改后为0）代表0 11代表1  这个不能用别的数字代替  不然统计就不对了 因为第一次统计的1肯定个数就不够呀  01代表2  10代表3 
         for(int i=0;i<row;i++){
            for(int j=0;j<col;j++)
            {
                int value = 0;
                //上
                if(i>0)
                {
                    value += (board[i-1][j]==1||board[i-1][j]==3)?1:0;
                }
                //下
                if(i<row-1)
                {
                    value += (board[i+1][j]==1||board[i+1][j]==3)?1:0;
                }
                //左
                if(j>0)
                {
                    value += (board[i][j-1]==1||board[i][j-1]==3)?1:0;
                }
                //右
                if(j<col-1)
                {
                    value += (board[i][j+1]==1||board[i][j+1]==3)?1:0;
                }
                //左上 
                if(j>0&&i>0)
                {
                    value += (board[i-1][j-1]==1||board[i-1][j-1]==3)?1:0;
                }
                //左下
                if(j>0&&i<row-1)
                {
                     value += (board[i+1][j-1]==1||board[i+1][j-1]==3)?1:0;
                }
                //右上
                if(i>0&&j<col-1)
                {
                     value += (board[i-1][j+1]==1||board[i-1][j+1]==3)?1:0;
                }
                //右下
                if(i<row-1&&j<col-1)
                {
                     value += (board[i+1][j+1]==1||board[i+1][j+1]==3)?1:0;
                }

                if(board[i][j]==1)
                {
                 if(value<2||value>3) board[i][j]=3;//1-0 
                else if(value==3||value==2) board[i][j]=1;
                }
                else
                {
                    if(value==3) board[i][j]=2;
                }
            }
        }
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++)
            {
                if(board[i][j]==3) board[i][j] = 0; 
                else if(board[i][j]==2) board[i][j] = 1;
            }}
    }
}

```
### 300 最长上升子序列
#### 利用dp算法来解决 cell数组用于存储此节点的上升子序列最大值 利用双层遍历即可解决
```
class Solution {
    public int lengthOfLIS(int[] nums) {
       if(nums.length==0) return 0;
        int[] cell = new int[nums.length];
        for(int i=0;i<nums.length;i++)
        {
            cell[i] = 1;//初始化为1
        }
        for(int i=0;i<nums.length;i++){
            for(int j=i;j>=0;j--)
            {
                if(nums[j]<nums[i])
                cell[i] = Math.max(cell[i],cell[j]+1);
            }
        }
        return Arrays.stream(cell).max().getAsInt();
    }
}
```
### 322 零钱兑换 dp算法
```
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[]  dp = new int[amount+1];
        dp[0] = 0;
        for(int i=1;i<=amount;i++)
        {
            dp[i] = amount+1;
        }
         for(int i=1;i<=amount;i++){
             for(int j=0;j<coins.length;j++)
             {
                 //如果此处可以用j来表示
                 if(coins[j]<=i)
                 dp[i] = Math.min(dp[i],dp[i-coins[j]]+1);
             }
         }
         return dp[amount]<=amount?dp[amount]:-1;
    }
}
```
### 加法 （不可利用加法，利用位运算）
```
同样我们可以用三步走的方式计算二进制值相加： 5---101，7---111

第一步：相加各位的值，不算进位，得到010，二进制每位相加就相当于各位做异或操作，101^111。
第二步：计算进位值，得到1010，相当于各位进行与操作得到101，再向左移一位得到1010，(101&111)<<1。
第三步重复上述两步，各位相加 010^1010=1000，进位值为100=(010 & 1010)<<1。
继续重复上述两步：1000^100 = 1100，进位值为0，跳出循环，1100为最终结果。
结束条件：进位为0，即a为最终的求和结果。
class Solution {
    public int getSum(int a, int b) {
        while(b!=0){
            int temp = a^b;
            b = (a&b)<<1;
            a = temp;
        }
        return a;
    }
}
```
### 350 两个数组的交集
### 先排序+双指针
```
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        ArrayList<Integer> arr = new ArrayList<>();
        for(int i=0,j=0;i<nums1.length&&j<nums2.length;)
        {
            if(nums1[i]==nums2[j])
            {
                arr.add(nums1[i]);
                i++;
                j++;
            }
            else if(nums1[i]>nums2[j])
            {
                j++;
            }
            else
            {
                i++;
            }
        }
       int [] result = new int[arr.size()];
       int e = 0;
       for(int i:arr)
       {
           result[e++] = i;
       }
       return result;
    }
}
```
### 324 摆动排序II
#### 先对数组进行排序 然后从后向前进行穿插
```
class Solution {
    public void wiggleSort(int[] nums) {
        int length = nums.length;
        Arrays.sort(nums);
        int[] temp = new int[length];
        for(int i=0;i<length;i++){
            temp[i] = nums[i];
        }
        int k = length-1;
        for(int i=1;i<length;i+=2){
            nums[i] = temp[k--];
        }
        for(int i=0;i<length;i+=2){
            nums[i] = temp[k--];
        }
    }
}
```
### 328 奇偶链表
#### 原地翻转 注意前后处理的逻辑关系 注意保存中间的节点
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if(head==null||head.next==null) return head;
        ListNode pre = head;
        ListNode cur = head.next;
        while(cur!=null&&cur.next!=null)
        {
            ListNode temp = pre.next;
            pre.next = cur.next;
            cur.next = cur.next.next;
            pre.next.next = temp;
            pre = pre.next;
            cur = cur.next;
        }
        return head;
    }
}
```
### 334 递增的三元序列
```
class Solution {
    public boolean increasingTriplet(int[] nums) {
        //双指针实现  无需连续
        int p1 = Integer.MAX_VALUE;
        int p2 = Integer.MAX_VALUE;
        for(int m:nums){
            if(m<=p1) p1 = m;
            else if(m<=p2) p2 = m;
            else  return true;
        }
        return false;
    }
}
```
### 341 扁平化嵌套
#### 仔细阅读接口函数 递归调用即可 
```
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     public boolean isInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     public Integer getInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     public List<NestedInteger> getList();
 * }
 */
public class NestedIterator implements Iterator<Integer> {
    Queue<Integer> res = new LinkedList<>();
    public NestedIterator(List<NestedInteger> nestedList) {
        get(nestedList);
    }
    public void get (List<NestedInteger> nestedList)
    {
        for(NestedInteger temp:nestedList)
        {
            if(temp.isInteger())
            res.offer(temp.getInteger());
            else
            get(temp.getList());
        }
    }

    @Override
    public Integer next() {
        return res.poll();
    }

    @Override
    public boolean hasNext() {
        return !res.isEmpty();
    }
}

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.hasNext()) v[f()] = i.next();
 */

```
### 395 至少有k个重复字符的最长字串
#### 分治法
```
class Solution {
    public int longestSubstring(String s, int k) {
        return help(s,k,0,s.length()-1);//left为左边的入口，right为字符串最后一个下标
    }
    public int help(String s,int k,int left, int right)
    {
        if(left>right)
        return 0;
        int[] cnt = new int[26];
        for(int i=0;i<26;i++)
        cnt[i] = 0;//初始化 
        for(int index=left;index<=right;index++)
        {
            cnt[s.charAt(index)-'a']++;
        }
        //修改left位置
        while(left<=right&&cnt[s.charAt(left)-'a']<k)
        {
            cnt[s.charAt(left)-'a']--;
            left++;
        }
        //修改right位置
         while(left<=right&&cnt[s.charAt(right)-'a']<k)
        {
            cnt[s.charAt(right)-'a']--;
            right--;
        }
        //寻找最大的值
        for(int index=left+1;index<right;index++)
        {
            if(cnt[s.charAt(index)-'a']<k)//如果有一点字符的出现次数不满足k次，则结果在他两边，利用分治解决
            return Math.max(help(s,k,left,index-1),help(s,k,index+1,right));

        }
        return right-left+1;
    }
}
```
### 454 四数相加
#### 正负相加 利用hashmap  map.getOrDefault(key,default) 如果map中有这个key,则返回对应的value 否则返回default
```
class Solution {
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        Map<Integer,Integer> map = new HashMap<>();
        int count = 0;
        for(int i=0;i<A.length;i++)
        {
            for(int j=0;j<B.length;j++)
            {
                map.put(A[i]+B[j],map.getOrDefault(A[i]+B[j],0)+1);
            }
        }
        for(int i=0;i<C.length;i++)
        {
            for(int j=0;j<D.length;j++)
            {
               count += map.getOrDefault(-C[i]-D[j],0);
            }
        }
        return count;
}}
```
### 滑动窗口最大值
#### Way1： 暴力解法 创建新数组存取res，之后利用两层for循环逐一挪动窗口并进行判断，将此窗口的最值插入res中即可。 O（n*k）k为windows大小
### 寻找两个有序数组的中位数 要求o(log(m+n))
#### Way1:merge两个数组 之后排序 最后直接取中位数即可。 O((m+n)log(m+n))
#### Way2:利用两个指针，分别指向两个数组的head，然后挪动到中位数即可。O((m+n)/2)
#### Way3: 一看见log的复杂度就要想到分治算法,此问题可以转化为min k问题，即寻找数组中第k小的元素。
最后从medianof two sorted arrays中看到了一种非常好的方法
该方法的核心是将原问题转变成一个寻找第k小数的问题（假设两个原序列升序排列），这样中位数实际上是第(m+n)/2小的数。所以只要解决了第k小数的问题，原问题也得以解决。

首先假设数组A和B的元素个数都大于k/2，我们比较A[k/2-1]和B[k/2-1]两个元素，这两个元素分别表示A的第k/2小的元素和B的第k/2小的元素。这两个元素比较共有三种情况：>、<和=。如果A[k/2-1]<B[k/2-1]，这表示A[0]到A[k/2-1]的元素都在A和B合并之后的前k小的元素中。换句话说，A[k/2-1]不可能大于两数组合并之后的第k小值，所以我们可以将其抛弃。

证明也很简单，可以采用反证法。假设A[k/2-1]大于合并之后的第k小值，我们不妨假定其为第（k+1）小值。由于A[k/2-1]小于B[k/2-1]，所以B[k/2-1]至少是第（k+2）小值。但实际上，在A中至多存在k/2-1个元素小于A[k/2-1]，B中也至多存在k/2-1个元素小于A[k/2-1]，所以小于A[k/2-1]的元素个数至多有k/2+ k/2-2，小于k，这与A[k/2-1]是第（k+1）的数矛盾。

当A[k/2-1]>B[k/2-1]时存在类似的结论。

当A[k/2-1]=B[k/2-1]时，我们已经找到了第k小的数，也即这个相等的元素，我们将其记为m。由于在A和B中分别有k/2-1个元素小于m，所以m即是第k小的数。(这里可能有人会有疑问，如果k为奇数，则m不是中位数。这里是进行了理想化考虑，在实际代码中略有不同，是先求k/2，然后利用k-k/2获得另一个数。)

通过上面的分析，我们即可以采用递归的方式实现寻找第k小的数。此外我们还需要考虑几个边界条件：

如果A或者B为空，则直接返回B[k-1]或者A[k-1]；
如果k为1，我们只需要返回A[0]和B[0]中的较小值；
如果A[k/2-1]=B[k/2-1]，返回其中一个；
最终实现的代码为：
```
double findKth(int a[], int m, int b[], int n, int k)
{
    //always assume that m is equal or smaller than n
    if (m > n)
        return findKth(b, n, a, m, k);
    if (m == 0)
        return b[k - 1];
    if (k == 1)
        return min(a[0], b[0]);
    //divide k into two parts
    int pa = min(k / 2, m), pb = k - pa;
    if (a[pa - 1] < b[pb - 1])
        return findKth(a + pa, m - pa, b, n, k - pa);
    else if (a[pa - 1] > b[pb - 1])
        return findKth(a, m, b + pb, n - pb, k - pb);
    else
        return a[pa - 1];
}
 
class Solution
{
public:
    double findMedianSortedArrays(int A[], int m, int B[], int n)
    {
        int total = m + n;
        if (total & 0x1)  //若为奇数则total最后一位二进制为1
            return findKth(A, m, B, n, total / 2 + 1);
        else
            return (findKth(A, m, B, n, total / 2)
                    + findKth(A, m, B, n, total / 2 + 1)) / 2;
    }
};
```
我们可以看出，代码非常简洁，而且效率也很高。在最好情况下，每次都有k一半的元素被删除，所以算法复杂度为logk，由于求中位数时k为（m+n）/2，所以算法复杂度为log(m+n)。
##### C++ code:
```
class Solution {
public:
    vector<int> trans(vector<int> &num,int p) //注意vector当作形参的时候不是地址，地址应该为&a[0] ,而此处递归需要传入一个vector对象，不是地址，注意与数组的区别。此函数用于构造转换的vector。
    {
        vector<int> res;
        for(int i=p;i<num.size();i++)
        {
            res.push_back(num[i]);
        }
        return res;
    }
int findk(vector<int> nums1,int m, vector<int> nums2,int n, int k)//用于寻找最小的K元素
{
        if(m>n)
            return findk(nums2,n,nums1,m,k);
        if(m==0)
            return nums2[k-1];
        if(k==1)
            return min(nums1[0],nums2[0]);
        int pa = min(k/2,m),pb = k-pa;
        if(nums1[pa-1]<nums2[pb-1])
        {vector<int> nums3 = trans(nums1,pa);
          return    findk(nums3,m-pa,nums2,n,k-pa);}
        if(nums1[pa-1]>nums2[pb-1])
        {
            vector<int> nums4 = trans(nums2,pb);
             return findk(nums1,m,nums4,n-pb,k-pb);
        }
           
        else
            return nums1[pa-1];
    }
    
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
       int m = nums1.size();
       int n = nums2.size();
        int total = nums1.size()+nums2.size();
        if(total & 0x1)  
            return double(findk(nums1,m,nums2,n,total/2+1));
        else
            return 0.5*(double(findk(nums1,m,nums2,n,total/2))+double(findk(nums1,m,nums2,n,(total/2)+1)));
    }
};
```
### 283 移动0
#### Way1:引入队列queue，遍历一遍数组，将非零的值按顺序存入，在遍历一遍，将其覆盖原来的数组内容，最后补0
 时间复杂度o(n) 空间复杂度o(n)
#### Way2:双指针法  slow指针保证前面的元素都是非0，fast向后遍历，一旦发现非零元素，就跟slow交换，之后slow++即可 （这样slow和fast之间全是0）
 时间复杂度o(n) 空间复杂度o(1)
### 27 移除元素
#### Way 此题目要求o(1)的空间复杂度，其本质上跟之前的移动0是一样的，只要把需要移除的元素都放到最后即可完成任务。同样采用双指针可以解决。
```
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int m = nums.size();
        if(m==0) return 0;
        int slow=0,fast=0;
        for(slow=0,fast=0;fast<m;fast++)
        {
            if(nums[fast]!=val)
                swap(nums[slow++],nums[fast]);
        }
    return slow;
    }
};
```
### 66 加一
#### Way 加一关键问题在于考虑何时需要进位，一个数加一需要进位的话就是所有的数都是9，只要有不是9的，则此为加1，其后面的数都为0即可。则按照此思路只需要从后向前遍历依次即可。时间复杂度O(n)  空间o(1) 
### 88 合并两个有序数组
#### Merge与之前的寻找两个有序数组的中位数类似，此外也是归并排序中的核心步骤，引入新的辅助数组和双指针，即可完成merge 
时间复杂度 o(n) 空间复杂度o(n)
### 268 缺失数字
#### Way1 利用等差数列求和得到总和 然后遍历数组即可得到结果 
时间复杂度 o(n)  空间o(1)
#### Way2  排序后跟下标进行对比 不同则返回下标即可
此方法不满足复杂度要求 最快的快排也只有o(nlogn)
### 169 求众数
#### Way1 排序后取中位数即可
#### Way2 利用hashmap来记录每个key出现的次数，记为value,之后选最大的value即可 
### 118 杨辉三角
* 主要就是抓住每一层的规律，其每层首尾都为1，其余部分都是由上一层算出来的，注意单独处理第一层即可。
```
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> result;
        for(int i=0;i<numRows;i++)
        {
            vector<int> temp;
            temp.push_back(1);
            if(i==0) 
            {     result.push_back(temp);
                continue;}
            for(int j=0;j<i-1;j++) //注意终止条件的判断
            {
                temp.push_back(result[i-1][j]+result[i-1][j+1]);
            }
            temp.push_back(1);
            result.push_back(temp);
        }
        return result;
    }
}; 
```
### 73 矩阵置零
#### 此题最简单就是引入o(mn)的空间复杂度，保存mn为0的点，之后遍历新矩阵，每当遇到0就将其对应的行、列置零；  其次也可以引入0(m+n)的空间复杂度，利用两个一维数组存储含0的行和列，之后遍历一遍数组，如果行或者列为0则将其置零； 最后可以引入O（1）的复杂度用第一行和第一列来标记是否有0，引入两个boolean变量标记第一行和第一列本身是否含0  之后根据这个结果对从（1,1）开始的地方进行遍历，将除了第一行第一列的东西做完，最后根据booleaan变量进行第一行和第一列的设置。
```
class Solution {
    public void setZeroes(int[][] matrix) {
        if(matrix.length==0) return;
        boolean firstrow = false;
        boolean firstcol = false;
        for(int i=0;i<matrix.length;i++)
        {
            for(int j=0;j<matrix[0].length;j++)
            {
                if(i!=0 && j!=0 && matrix[i][j]==0)
                {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
                else
                {
                    if(i==0 && matrix[i][j]==0)
                    firstrow = true;
                    if(j==0 && matrix[i][j]==0)
                    firstcol = true;
                }
            }
        }
        for(int i=1;i<matrix.length;i++)
        {
            for(int j=1;j<matrix[0].length;j++)
            {
                if(matrix[i][0]==0||matrix[0][j]==0)
                matrix[i][j] = 0;
            }
        }
            if(firstcol){
                for(int i=0;i<matrix.length;i++){
                    matrix[i][0] = 0;
                }
            }
            if(firstrow){
                for(int i=0;i<matrix[0].length;i++){
                    matrix[0][i] = 0;
                }
            }

    }
}
```
## 链表部分：
### 237 删除链表中的节点
#### 此题特殊在于函数传入的是要删除的节点的指针，所以无法找到prev而进行删除工作，而题目只要求val删除，考虑将后面的元素挪过来，然后将其覆盖。
### 21 合并两个有序链表
#### Way1 递归
```
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1==NULL)
            return l2;
        else if(l2==NULL)
            return l1;
        else if(l1->val<=l2->val)
        {  l1->next = mergeTwoLists(l1->next,l2);
            return l1;}
        else
        {l2->next = mergeTwoLists(l1,l2->next);
        return l2;}
    }
};
```
#### Way2 类比归并排序的merge来做，数组的话需要辅助数组，此处需要辅助指针，两个指针一个用来遍历所有的链表，一个记住链表头部用于返回。
```
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode * q = new ListNode(0);
    ListNode * p = q;  注意设置两个指针的含义
        if(l1==NULL) return l2;
        if(l2==NULL) return l1;
        while(l1!=NULL && l2!=NULL)
        {
            if(l1->val<=l2->val)
               {p->next = l1;
                l1 = l1->next;}
            else
            {
                p->next = l2;
                l2 = l2->next;
                
            }
            p = p->next;
         }
        if(l1==NULL)
            p->next = l2;
        else if(l2==NULL)
            p->next = l1;
       return q->next; 
     }
};
```
### 206 反转链表
#### Way1:迭代完成
利用两个指针完成，迭代完成，一个指针不断移动，一个指针一直标记新反转的头，时间复杂度o(n),空间复杂度o(1).
```
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode*p = NULL;//p一直标记新反转的链表的头
        ListNode* q = head;//不断移动修改链表
        while(q!=NULL)
        {
            ListNode* temp = q->next;
             q->next = p;
             p = q;
             q = temp;
        }
        return p;
    }
};
```
#### Way2：递归
```
 public ListNode reverseList(ListNode head) {
     if(head==null) {
		   return null;
	   }  
	ListNode last=head;
	ListNode current=last.next;
	  
	   return reverse(head,last,current);
    }
  private ListNode reverse(ListNode head, ListNode last, ListNode current) {
	if(current==null) {
		return head;
	}
	last.next=current.next;
	ListNode temp=current.next;
	current.next=head;
	//注意此时reverse方法的参数
	return reverse(current,last,temp);
	
}
```
### 23 合并K个排序链表
#### 
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<> (new Comparator<ListNode>() {
            public int compare(ListNode l1,ListNode l2){
                    return l1.val-l2.val;
                    //优先队列默认是最小堆（comparator的方法l1-l2） 重写 comparator方法是写对比的规则（l2-l1则为最大堆） 此处重写只是为了改变判断的内容 为链表的val而不是节点
                    //
            }
          
        });
        for(ListNode l:lists)//遍历链表 将各个表的表头放入优先队列 
        {
            if(l!=null)
            pq.add(l);
        }
        ListNode head = new  ListNode(0); //新的表头
        head.next = null;
        ListNode tail = head;
        while(!pq.isEmpty())
        {
            tail.next = pq.poll();//拿到队列里最小的元素  
            tail = tail.next; 
            if(tail.next!=null)
             pq.add(tail.next);//将拿走的元素的下一个元素放入queue
             tail.next = null;//将此新节点的next为null
        }
        return head.next;
    }
}
```
### 234 回文链表
#### Way1 对链表进行计数，将其一半压入栈中，然后与另一部分进行对比，即可。
时间复杂度o(n) 空间复杂度o(n)
#### Way2 为了降低空间复杂度，利用快慢指针找到链表终点，之后将链表后半部分反转，最后判断两个链表是否相等。
```
class Solution {
public:
    ListNode* findmiddle(ListNode* head){
        ListNode* slow = head;
        ListNode* fast = head->next;
        while(fast!=NULL && fast->next!=NULL)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
    ListNode * reverse(ListNode * head)
    {
    ListNode* p = NULL;
    ListNode* q = head;
        while(q!=NULL)
    {
      ListNode* temp = q->next;
            q->next = p;
            p = q;
            q = temp;
    }
        return p;
    }
    bool isPalindrome(ListNode* head) {
        //先找到链表中间，slow指向第一个，fast指向第二个，若为奇数，slow会停在最中间，若为偶数会停在1/2左边。
        if(head==NULL||head->next==NULL) return true;
        ListNode* mid = findmiddle(head);
        mid = mid->next;
        mid = reverse(mid);
        while(mid&&head){
        if(mid->val!=head->val)
            return false;
        mid = mid->next;
        head = head->next;
        }
        return true;
    }
};
```
### 160 相交链表
#### Way1:固定a链表，遍历b链表，如果有相同的next指针则说明相交点。
时间复杂度o(n²)
#### Way2:hash法 
遍历链表 A 并将每个结点的地址/引用存储在哈希表中。然后检查链表 B 中的每一个结点 b是否在哈希表中。若在，则为相交结点。
时间复杂度 : O(m+n) 
空间复杂度 : O(m)或O(n)。
#### Way3:双指针法 非常巧妙 都遍历一遍如果有相交点必定会相遇
```
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA==NULL||headB==NULL)
            return NULL;
        ListNode* a = headA;
        ListNode* b = headB;
        while(a!=b)
        {
            a=(a==NULL)?headB:a->next;
            b=(b==NULL)?headA:b->next;
        }
    return a;
    }
```
### 141 环形链表
#### way1:考虑快慢指针法，如若有环，则一定回相遇，可以类比物理中的追击问题，如果fast指针已经走到链表结尾，则说明没有环，因为有环的话也不会结束的。
时间复杂度o(n),空间复杂度o(1)
要注意编程的时候程序的`健壮性`，考虑特殊情况下程序的结果。
还有就是指针是否有效的问题，因为要访问`fast->next->next`,所以需要注意while的判断条件。
```
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(head==NULL||head->next==NULL) return false;
        ListNode *slow = head;
        ListNode* fast = head->next;
        while(fast!=NULL&&fast->next!=NULL){
          if(slow==fast)
              return true;
         else{
             slow = slow->next;
             fast = fast->next->next;    
         }
        }
        return false;
    }
};
```
## 19 删除链表的倒数第N个节点
注意哑节点的使用（不然无法应对特殊情况，比如删除第一个，n+1就会超出范围）
此处使用双指针 只需遍历一次即可。
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode res = new ListNode(0);
        res.next = head;
        ListNode  curr = res, p = res;
        for(int i=0;i<=n;i++)
            p = p.next;
    while(p!=null)
    {
        p = p.next;
        curr = curr.next;
    }
        curr.next = curr.next.next;
    return res.next;
    }
}

```
## 栈（此后语言改为JAVA）
### 20 有效的括号
#### way1:此题为栈的典型应用，由于刚改为java刷题，很多小问题需要注意。 
判断是否为有效括号首先可以这样想，一旦有个左括号，如果下一个符号不是右括号，则需要等待，若为右括号，则需要一定与它匹配，也就是说，核心问题是一个最近匹配问题。所以我们如果采用栈，遍历整个字符串，有左括号则压进去，有右括号则出栈，若匹配则继续向下遍历，若不匹配则不对(还可能出现 如果右括号多 栈空则也是不匹配  左括号多遍历完栈不为空也不匹配)  此外要注意 map的使用方法。<br>
时间复杂度 O（n） 空间复杂度o(n)
```
class Solution {
    public boolean isValid(String s) {
        int len = s.length();
        if(len==0 ) //特殊情况算匹配
            return true;
        if((len%2)!=0) //若为奇数则直接退出
            return false;
        HashMap map = new HashMap(); //一定是key,value
        map.put(')','(');
        map.put('}','{');
        map.put(']','[');
        Stack<Character> stack = new Stack(); 
        for(char val:s.toCharArray()) //注意这种for循环以及string类的一些方法
	{
           if(val=='('||val=='{'||val=='[')
            stack.push(val);
            if((val==')'||val=='}'||val==']')&&(stack.isEmpty()||stack.pop()!=map.get(val))) //栈提前空或者不匹配都是错的
                return false;
        }
        return stack.isEmpty();//巧妙转换 遍历完栈应该为空才说明正常 如不是则说明左括号多
    }
}
```
### 155最小栈
#### way1：同步辅助栈
采用利用空间换时间的方法，题目要求getMin在常数时间内返回，则需额外存储。不然无法完成。采用两个栈完成，一个用来存储数据，一个用来标注最小的数。
`时间复杂度o(1)`  `空间复杂度o(n)`
```
class MinStack {
    private Stack<Integer> data;//数据栈
    private Stack<Integer> min;//辅助栈
    /** initialize your data structure here. */
    public MinStack() {
        this.data = new Stack<Integer>();//初始化
        this.min = new Stack<Integer>();
    }
    
    public void push(int x) {
        this.data.push(x);
        if(this.min.isEmpty()||this.getMin()>=x) //注意辅助栈不是存储一个数据，需要考虑刚开始以及后面要用等号（考虑有多个相同最小值）
            this.min.push(x);
        }
    
    public void pop() {
        if(this.data.isEmpty()) //如果栈空则抛出异常
            throw new RuntimeException("your stack is empty!");
       int val = this.data.pop();//注意判断出栈的是否是最小的元素，不是的话不用删辅助栈里面的
       if(val == this.getMin())
           this.min.pop();
    }
    
    public int top() {
        if(this.data.isEmpty())
            throw new RuntimeException("your stack is empty!");
        return this.data.peek(); //注意peek不会删除元素，pop会
    }
    
    public int getMin() {
        if(this.min.isEmpty())  
            throw new RuntimeException("your stack is empty!");
        return this.min.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```
## 二叉树
### 101 对称二叉树（镜像二叉树）
#### way1:递归解法
此题本质上就是一个对称问题，满足对称二叉树的要求无非就是以中间为线可以对折，那么需要满足的就是1)根节点val相同 2）每个树的右子树都与另一个树的左子树镜像对称。
`时间复杂度0(n)` `空间复杂度o(n)`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return judge(root,root);
    }
        public boolean judge(TreeNode r1,TreeNode r2){
            if(r1==null&&r2==null) return true;
            if(r1==null||r2==null) return false;
            return (r1.val==r2.val)&&judge(r1.left,r2.right)&&judge(r1.right,r2.left);
        
        }
}
//注意此方法好理解 
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null) return true; //判断根节点
        return judge(root.left,root.right);  
    }
        public boolean judge(TreeNode r1,TreeNode r2){
            if(r1==null&&r2==null) return true;
            if(r1==null||r2==null) return false;
            return (r1.val==r2.val)&&judge(r1.left,r2.right)&&judge(r1.right,r2.left);
        
        }
}
```
#### way2:迭代解法
利用层次遍历的方法对二叉树进行遍历，注意入队列的顺序按照镜像来。
`时间复杂度o（h)`  `空间复杂度o(n)`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
       if(root==null) return true;
        Queue<TreeNode> c =new LinkedList<>();
        c.add(root.left);
        c.add(root.right);
        while(!c.isEmpty()){
            TreeNode t1 = c.poll();
            TreeNode t2 = c.poll();
            if(t1==null&&t2==null) continue;
            if(t1==null||t2==null||(t1.val!=t2.val)) return false;
            c.add(t1.left);
            c.add(t2.right);
            c.add(t1.right);
            c.add(t2.left);    
          }
        return true;
    }
}
```
### 108 将有序数组转为二叉平衡搜索树
#### way1 递归解法
此题核心在于搜索二叉树的中序遍历一定是一个有序的数组，所以此题目核心便是一个按照中序遍历的思想去构造二叉树。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return   balance(nums,0,nums.length-1);
    }
   private  TreeNode  balance(int[] nums,int l,int r){
       if(l>r) return null; //注意边界条件一定是l>r，相等时不可退出 需要将其赋值进去。
       int mid = (l+r)>>>1;// 注意中点利用（l+r）/2可能溢出（可能回超过int的表示范围），此方法为java JDK内部实现，不会溢出。也可以采用l+（r-l)/2
       TreeNode root = new TreeNode(nums[mid]);
       root.left = balance(nums,l,mid-1); 
       root.right = balance(nums,mid+1,r);
       return root;
   }
}
```
### 104 求二叉树的深度
way1:递归 
利用dfs遍历一遍即可。
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null)
            return 0;
        else
        {
         int left = maxDepth(root.left);
        int right = maxDepth(root.right);
         return Math.max(left,right)+1;    
        }
    }
}
```
## 70 爬楼梯
### way1 :此题本质上是斐波拉契数列的问题，考虑动态规划的思想 总方法应该是n-1与n-2的和，则找到递推公式dp[n]=dp[n-1]+dp[n-2]
```
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n+1];
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=n;i++)
            dp[i]= dp[i-1]+dp[i-2];
    return dp[n];
    }
}
```
### way2:利用通项公式（可能会损失精度）
```
class Solution {
    public int climbStairs(int n) {
        double sqrt_5 = Math.sqrt(5);
        double fib_n = Math.pow((1 + sqrt_5) / 2, n + 1) - Math.pow((1 - sqrt_5) / 2,n + 1);
        return (int)(fib_n / sqrt_5);
    }
}
```
## 344 翻转字符串（注意空间复杂度必须是o(1))
### way1 利用双指针法 
```
class Solution {
    public void reverseString(char[] s) {
        if(s.length==0) return;
        int left = 0;
        int right = s.length-1;
        while(left<=right)
        {
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;
            left++;
            right--;
        }
    }
}
```
