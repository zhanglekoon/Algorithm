#Leetcode笔记 精选面试top145
#数组部分：

1.两数之和   返回和为target的两个元素的下标
方法一：暴力法  时间复杂度o(n2)  没有多余开辟的空间
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
方法二：hash map 利用hash以空间换取时间的思想 时间复杂度为o(n) 但是要考虑如果hash冲突的问题  
可以从这个角度考虑 不要先将所有的值都加入hash map中去，每个元素加进去的必要条件是当前这个元素和hash map中的元素没有sum为target的值，如果有的话直接返回（即如果是两个下标不同，但value相同的元素需要采取此办法）。
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
2.最大连续子序列和
这是一个典型的动态规划问题，考虑temp保存当前最大的连续自序和，但要考虑temp保存的恰是以子序列的结束点为基准，所以递推关系式为temp = max(temp+num[i],num[i]) 此处的最大和要么是跟上一个子序列相加得到的，要么就是自己。 之后res用来保存最大的temp即是结果。
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
3.买卖股票的最佳时机  (找到最大的山谷和山峰的区间)
我们需要找出给定数组中两个数字之间的最大差值（即，最大利润）。此外，第二个数字（卖出价格）必须大于第一个数字（买入价格）。
形式上，对于每组i和 j（其中 j > i）我们需要找出max(prices[j]−prices[i])。
方法一：暴力法
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
复杂度分析
时间复杂度：O(n^2)
空间复杂度：O(1)

方法二：一次遍历
假设给定的数组为：[7, 1, 5, 3, 6, 4]
如果我们在图表上绘制给定数组中的数字，我们将会得到一个函数图
使我们感兴趣的点是上图中的峰和谷。我们需要找到最小的谷之后的最大的峰。
我们可以维持两个变量——minprice 和 maxprofit，它们分别对应迄今为止所得到的最小的谷值和最大的利润（卖出价格与最低价格之间的最大差值）
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
复杂度o(n)
4.买卖股票的最佳时机II（可以多次买卖，让利润最大）
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
一次遍历：复杂度o(n)
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
5.存在重复元素
如果数组中有出现两次以上的元素，则返回true,否则返回false
方法1：调用sort排序,之后遍历，一旦发现有相同的元素直接return
复杂度o(nlogn)
方法2：set 将元素放入set中，每次放进去的时候进行对比。
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
方法3：hash表

6.翻转数组
考虑利用%进行翻转，翻转k次的结果等于l=k%n次。
首先翻转所有的数组，之后翻转l的部分，最后翻转n-l的部分即可
时间复杂度o(n) 空间复杂度o(1)
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

7.滑动窗口最大值
Way1： 暴力解法 创建新数组存取res，之后利用两层for循环逐一挪动窗口并进行判断，将此窗口的最值插入res中即可。 O（n*k）k为windows大小
8.寻找两个有序数组的中位数 要求o(log(m+n))
Way1:merge两个数组 之后排序 最后直接取中位数即可。 O((m+n)log(m+n))
Way2:利用两个指针，分别指向两个数组的head，然后挪动到中位数即可。O((m+n)/2)
Way3: 一看见log的复杂度就要想到分治算法,此问题可以转化为min k问题，即寻找数组中第k小的元素。
最后从medianof two sorted arrays中看到了一种非常好的方法。原文用英文进行解释，在此我们将其翻译成汉语。该方法的核心是将原问题转变成一个寻找第k小数的问题（假设两个原序列升序排列），这样中位数实际上是第(m+n)/2小的数。所以只要解决了第k小数的问题，原问题也得以解决。

首先假设数组A和B的元素个数都大于k/2，我们比较A[k/2-1]和B[k/2-1]两个元素，这两个元素分别表示A的第k/2小的元素和B的第k/2小的元素。这两个元素比较共有三种情况：>、<和=。如果A[k/2-1]<B[k/2-1]，这表示A[0]到A[k/2-1]的元素都在A和B合并之后的前k小的元素中。换句话说，A[k/2-1]不可能大于两数组合并之后的第k小值，所以我们可以将其抛弃。

证明也很简单，可以采用反证法。假设A[k/2-1]大于合并之后的第k小值，我们不妨假定其为第（k+1）小值。由于A[k/2-1]小于B[k/2-1]，所以B[k/2-1]至少是第（k+2）小值。但实际上，在A中至多存在k/2-1个元素小于A[k/2-1]，B中也至多存在k/2-1个元素小于A[k/2-1]，所以小于A[k/2-1]的元素个数至多有k/2+ k/2-2，小于k，这与A[k/2-1]是第（k+1）的数矛盾。

当A[k/2-1]>B[k/2-1]时存在类似的结论。

当A[k/2-1]=B[k/2-1]时，我们已经找到了第k小的数，也即这个相等的元素，我们将其记为m。由于在A和B中分别有k/2-1个元素小于m，所以m即是第k小的数。(这里可能有人会有疑问，如果k为奇数，则m不是中位数。这里是进行了理想化考虑，在实际代码中略有不同，是先求k/2，然后利用k-k/2获得另一个数。)

通过上面的分析，我们即可以采用递归的方式实现寻找第k小的数。此外我们还需要考虑几个边界条件：

如果A或者B为空，则直接返回B[k-1]或者A[k-1]；
如果k为1，我们只需要返回A[0]和B[0]中的较小值；
如果A[k/2-1]=B[k/2-1]，返回其中一个；
最终实现的代码为：

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
我们可以看出，代码非常简洁，而且效率也很高。在最好情况下，每次都有k一半的元素被删除，所以算法复杂度为logk，由于求中位数时k为（m+n）/2，所以算法复杂度为log(m+n)。
C++ code:

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
283 移动0
Way1:引入队列queue，遍历一遍数组，将非零的值按顺序存入，在遍历一遍，将其覆盖原来的数组内容，最后补0
此方法 时间复杂度o(n) 空间复杂度o(n)
Way2:双指针法  slow指针保证前面的元素都是非0，fast向后遍历，一旦发现非零元素，就跟slow交换，之后slow++即可 （这样slow和fast之间全是0）
此方法 时间复杂度o(n) 空间复杂度o(1)
27 移除元素
Way 此题目要求o(1)的空间复杂度，其本质上跟之前的移动0是一样的，只要把需要移除的元素都放到最后即可完成任务。同样采用双指针可以解决。
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
66 加一
Way 加一关键问题在于考虑何时需要进位，一个数加一需要进位的话就是所有的数都是9，只要有不是9的，则此为加1，其后面的数都为0即可。则按照此思路只需要从后向前遍历依次即可。时间复杂度O(n)  空间o(1) 
88 合并两个有序数组
Merge与之前的寻找两个有序数组的中位数类似，此外也是归并排序中的核心步骤，引入新的辅助数组和双指针，即可完成merge 
时间复杂度 o(n) 空间复杂度o(n)
268 缺失数字
Way1 利用等差数列求和得到总和 然后遍历数组即可得到结果 
时间复杂度 o(n)  空间o(1)
Way2  排序后跟下标进行对比 不同则返回下标即可
此方法不满足复杂度要求 最快的快排也只有o(nlogn)
169 求众数
Way1 排序后取中位数即可
Way2 利用hashmap来记录每个key出现的次数，记为value,之后选最大的value即可 
118 杨辉三角
主要就是抓住每一层的规律，其每层首尾都为1，其余部分都是由上一层算出来的，注意单独处理第一层即可。
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

链表部分：
237 删除链表中的节点
此题特殊在于函数传入的是要删除的节点的指针，所以无法找到prev而进行删除工作，而题目只要求val删除，考虑将后面的元素挪过来，然后将其覆盖。
21 合并两个有序链表
Way1 递归
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
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
Way2 类比归并排序的merge来做，数组的话需要辅助数组，此处需要辅助指针，两个指针一个用来遍历所有的链表，一个记住链表头部用于返回。
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
206 反转链表
Way1:迭代完成
利用两个指针完成，迭代完成，一个指针不断移动，一个指针一直标记新反转的头，时间复杂度o(n),空间复杂度o(1).
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
Way2：递归
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
234 回文链表
Way1 对链表进行计数，将其一半压入栈中，然后与另一部分进行对比，即可。
时间复杂度o(n) 空间复杂度o(n)
Way2 为了降低空间复杂度，利用快慢指针找到链表终点，之后将链表后半部分反转，最后判断两个链表是否相等。
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
160 相交链表
Way1:固定a链表，遍历b链表，如果有相同的next指针则说明相交点。
时间复杂度o(n²)
Way2:hash法 
遍历链表 A 并将每个结点的地址/引用存储在哈希表中。然后检查链表 B 中的每一个结点 b是否在哈希表中。若在，则为相交结点。
复杂度分析
时间复杂度 : O(m+n) 
空间复杂度 : O(m)或O(n)。
Way3:双指针法 非常巧妙 都遍历一遍如果有相交点必定会相遇
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
};
