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
### 暴力法求解无论是递归还是非递归都会有问题  栈溢出或者超时 所以需要采用的方法为奇偶分离  如果为偶数 xn=x(n/2)*x(n/2)如果为奇数，则为xn=x(n-1)/2 x(n-1)/2 x 
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
