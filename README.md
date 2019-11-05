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
