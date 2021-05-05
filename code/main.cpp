//#pragma GCC optimize("Ofast,unroll-loops")
//#pragma GCC target("avx2,tune=native")
#include <bits/stdc++.h>

using namespace std;

#define all(v) (v).begin(), (v).end()
#define sz(a) ((ll)(a).size())
#define X first
#define Y second

using ll = long long;
using ull = unsigned long long;
using dbl = long double;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
ll myRand(ll mod) {
    return (ull)rng() % mod;
}

const int inf = (int) 1e9;

namespace NCommon {
    
    struct Node {
        int left, right; // inclusive, from 0 to n - 1
        shared_ptr<Node> leftChild, rightChild;
        
        Node(int left, int right):
            left(left), right(right), leftChild(nullptr), rightChild(nullptr) {}
    };
    
    struct Segment {
        int left, right; // inclusive, from 0 to n - 1
        int cost;
    };
    
    void validate(shared_ptr<Node> root, int left, int right) {
        assert(root != nullptr);
        assert(root->left == left && root->right == right);
        if (left == right) {
            assert(root->leftChild == nullptr);
            assert(root->rightChild == nullptr);
        } else {
            assert(root->leftChild != nullptr);
            assert(root->rightChild != nullptr);
            int mid = root->leftChild->right;
            assert(left <= mid && mid < right);
            assert(root->rightChild->left == mid + 1);
            validate(root->leftChild, left, mid);
            validate(root->rightChild, mid + 1, right);
        }
    }
    
    int evaluate(shared_ptr<Node> root, const vector<Segment> &segments) {
        assert(root != nullptr);
        int result = 0;
        vector<Segment> activeSegments;
        for (auto seg: segments) {
            if (seg.left <= root->left && root->right <= seg.right) {
                result += seg.cost;
            } else if (!(seg.left > root->right || root->left > seg.right)) {
                result += seg.cost;
                activeSegments.push_back(seg);
            }
        }
        
        if (!activeSegments.empty()) {
            result += evaluate(root->leftChild, activeSegments);
            result += evaluate(root->rightChild, activeSegments);
        }
        return result;
    }
    
    vector<Segment> generate(int n, int m, int maxCost) {
        assert(n >= 1);
        
        vector<Segment> result;
        for (int i = 0; i < m; ++i) {
            int left = rand() % n;
            int right = left + rand() % (n - left);
            int cost = 1 + rand() % maxCost;
            assert(0 <= left && left <= right && right < n && cost > 0);
            result.push_back(Segment{left, right, cost});
        }
        return result;
    }
}

namespace NNaive {
    
    using Node = NCommon::Node;
    using Segment = NCommon::Segment;
    
    pair<int, shared_ptr<Node>> findSolution(
        const vector<Segment> &segments, int left, int right
    ) {
        auto root = make_shared<Node>(left, right);

        int sumCost = 0;
        vector<Segment> activeSegments;
        for (auto seg: segments) {
            if (seg.left <= root->left && root->right <= seg.right) {
                sumCost += seg.cost;
            } else if (!(seg.left > root->right || root->left > seg.right)) {
                sumCost += seg.cost;
                activeSegments.push_back(seg);
            }
        }
        
        if (left == right) {
            return make_pair(sumCost, root);
        }
        
        int best = inf;
        for (int mid = left; mid < right; ++mid) {
            auto [leftCost, leftChild] = findSolution(activeSegments, left, mid);
            auto [rightCost, rightChild] = findSolution(activeSegments, mid + 1, right);
            if (leftCost + rightCost < best) {
                best = leftCost + rightCost;
                root->leftChild = leftChild;
                root->rightChild = rightChild;
            }
        }
        
        return make_pair(best + sumCost, root);
    }
    
    int getCost(int n, const vector<Segment>& segments) {
        for (const auto &seg: segments) {
            assert(seg.cost > 0);
            assert(0 <= seg.left && seg.left <= seg.right && seg.right <= n - 1);
        }
        
        auto [resultCost, root] = findSolution(segments, 0, n - 1);
        NCommon::validate(root, 0, n - 1);
        int evaluationCost = NCommon::evaluate(root, segments);
        assert(resultCost == evaluationCost);
        return evaluationCost;
    }
}

namespace NCubic {
    
    using Node = NCommon::Node;
    using Segment = NCommon::Segment;
    
    shared_ptr<Node> constructNode(int left, int right, const vector<vector<int>>& opt) {
        auto node = make_shared<Node>(left, right);
        if (left != right) {
            int mid = opt[left][right];
            node->leftChild = constructNode(left, mid, opt);
            node->rightChild = constructNode(mid + 1, right, opt);
        }
        return node;
    }
    
    /*void calculateCostNaive(vector<vector<int>> &cost, const vector<Segment> &segments, int n) {
        // O(n * n * |segments|)
        for (int left = 0; left < n; ++left) {
            for (int right = left; right < n; ++right) {
               for (auto seg: segments) {
                   if (!(left > seg.right || seg.left > right)) {
                       cost[left][right] += seg.cost;
                   }
                   // left < right is not necessary as cost of leaf will count anyway
                   if (left < right && seg.left <= left && right <= seg.right) {
                       cost[left][right] -= 2 * seg.cost;
                   }
               }
            }
        }
    }*/
    
    void calculateCostQuadratic(vector<vector<int>> &cost, const vector<Segment> &segments, int n) {
        vector<vector<int>> exact(n, vector<int>(n));
        for (auto seg: segments) {
            exact[seg.left][seg.right] += seg.cost;
        }
        vector<vector<int>> include(n, vector<int>(n));
        for (int len = n; len >= 1; --len) {
            for (int left = 0; left + len <= n; ++left) {
                int right = left + len - 1;
                if (left == 0 && right == n - 1) {
                    include[left][right] = exact[left][right];
                } else if (left == 0) {
                    include[left][right] = include[left][right + 1] + exact[left][right];
                } else if (right == n - 1) {
                    include[left][right] = include[left - 1][right] + exact[left][right];
                } else {
                    include[left][right] = exact[left][right];
                    include[left][right] += include[left - 1][right];
                    include[left][right] += include[left][right + 1];
                    include[left][right] -= include[left - 1][right + 1];
                }
            }
        }
        
        vector<vector<int>> intersect(n, vector<int>(n));
        for (int left = 0; left < n; ++left) {
            intersect[left][left] = include[left][left];
            for (int right = left + 1; right < n; ++right) {
                // add doesn't depend on left, only on right
                int add = include[right][right] - include[right - 1][right];
                intersect[left][right] = intersect[left][right - 1] + add;
            }
        }
        
        for (int left = 0; left < n; ++left) {
            for (int right = left; right < n; ++right) {
               cost[left][right] = intersect[left][right] - 2 * include[left][right];
            }
        }
        
        /*for (int a = 0; a < n; ++a) {
            for (int b = a; b < n; ++b) {
                for (int c = b; c < n; ++c) {
                    for (int d = c; d < n; ++d) {
                        assert(cost[a][d] >= cost[b][c]);
                        // inverse quadrangle inequality
                        assert(cost[a][c] + cost[b][d] >= cost[a][d] + cost[b][c]);
                    }
                }
            }
        }*/
    }
    
    shared_ptr<Node> findSolution(
        const vector<Segment> &segments, int n
    ) {
        vector<vector<int>> cost(n, vector<int>(n));
        
        // calculateCostNaive(cost, segments, n);
        calculateCostQuadratic(cost, segments, n);
        
        vector<vector<int>> dp(n, vector<int>(n, inf));
        vector<vector<int>> opt(n, vector<int>(n, -1));
        for (int left = n - 1; left >= 0; --left) {
            dp[left][left] = cost[left][left];
            opt[left][left] = left;
            for (int right = left + 1; right < n; ++right) {
                for (int mid = left; mid < right; ++mid) {
                    int temp = cost[left][right] + dp[left][mid] + dp[mid + 1][right];
                    if (temp < dp[left][right]) {
                        dp[left][right] = temp;
                        opt[left][right] = mid;
                    }
                }
            }
        }
        return constructNode(0, n - 1, opt);
    }
    
    int getCost(int n, const vector<Segment>& segments) {
        for (const auto &seg: segments) {
            assert(seg.cost > 0);
            assert(0 <= seg.left && seg.left <= seg.right && seg.right <= n - 1);
        }
        
        auto root = findSolution(segments, n);
        NCommon::validate(root, 0, n - 1);
        int evaluationCost = NCommon::evaluate(root, segments);
        return evaluationCost;
    }
}

namespace NApprox {
    
    using Node = NCommon::Node;
    using Segment = NCommon::Segment;
    
    shared_ptr<Node> findSolutionBST(const vector<int> &w, int left, int right) {
        // can be done in O(n) but I'm too lazy now
        
        auto root = make_shared<Node>(left, right);
        if (left == right) {
            return root;
        }
        int mindiff = inf;
        int sum = accumulate(w.begin() + left, w.begin() + right, 0);
        int prefsum = 0;
        for (int i = left; i < right; ++i) {
            int diff = abs(prefsum - (sum - prefsum - w[i]));
            mindiff = min(mindiff, diff);
            prefsum += w[i];
        }
        prefsum = 0;
        for (int i = left; i < right; ++i) {
            int diff = abs(prefsum - (sum - prefsum - w[i]));
            if (diff == mindiff) {
                root->leftChild = findSolutionBST(w, left, i);
                root->rightChild = findSolutionBST(w, i + 1, right);
                return root;
            }
            prefsum += w[i];
        }
        assert(false);
    }
    
    shared_ptr<Node> findSolution(
        const vector<Segment> &segments, int n
    ) {
        // approximate [L, R] with sum of depths of queries [1; R] and [L; n - 1]
        vector<int> w(n - 1);
        for (auto seg: segments) {
            if (seg.right < n - 1) {
                w[seg.right] += seg.cost;
            }
            if (seg.left > 0) {
                w[seg.left - 1] += seg.cost;
            }
        }
        // dp[L][R] = (w[L] + ... + w[R - 1]) + min dp[L][m] + dp[m + 1][R]
        // now it is just optimal bst problem on w
        
        return findSolutionBST(w, 0, n - 1);
    }
    
    int getCost(int n, const vector<Segment>& segments) {
        for (const auto &seg: segments) {
            assert(seg.cost > 0);
            assert(0 <= seg.left && seg.left <= seg.right && seg.right <= n - 1);
        }
        
        auto root = findSolution(segments, n);
        NCommon::validate(root, 0, n - 1);
        int evaluationCost = NCommon::evaluate(root, segments);
        return evaluationCost;
    }
}

namespace NApproxExactBST {
    
    using Node = NCommon::Node;
    using Segment = NCommon::Segment;
    
    using NCubic::constructNode;
    
    shared_ptr<Node> findSolutionExactBST(const vector<int> &w, int n) {
        // TODO: knuth optimization
        
        assert(sz(w) == n - 1);
        
        vector<int> prefsum(n);
        for (int i = 1; i <= n - 1; ++i) {
            prefsum[i] = prefsum[i - 1] + w[i - 1];
        }
        auto getSum = [&prefsum] (int left, int right) {
            return prefsum[right + 1] - prefsum[left];
        };
        
        vector<vector<int>> dp(n, vector<int>(n, inf));
        vector<vector<int>> opt(n, vector<int>(n, -1));
        for (int left = n - 1; left >= 0; --left) {
            dp[left][left] = 0;
            opt[left][left] = left;
            for (int right = left + 1; right < n; ++right) {
                for (int mid = left; mid < right; ++mid) {
                    int temp = dp[left][mid] + dp[mid + 1][right];
                    if (temp < dp[left][right]) {
                        dp[left][right] = temp;
                        opt[left][right] = mid;
                    }
                }
                dp[left][right] += getSum(left, right - 1);
            }
        }
        return constructNode(0, n - 1, opt);
    }
    
    shared_ptr<Node> findSolution(
        const vector<Segment> &segments, int n
    ) {
        // approximate [L, R] with sum of depths of queries [1; R] and [L; n - 1]
        vector<int> w(n - 1);
        for (auto seg: segments) {
            if (seg.right < n - 1) {
                w[seg.right] += seg.cost;
            }
            if (seg.left > 0) {
                w[seg.left - 1] += seg.cost;
            }
        }
        // dp[L][R] = (w[L] + ... + w[R - 1]) + min dp[L][m] + dp[m + 1][R]
        // now it is just optimal bst problem on w
        
        return findSolutionExactBST(w, n);
    }
    
    int getCost(int n, const vector<Segment>& segments) {
        for (const auto &seg: segments) {
            assert(seg.cost > 0);
            assert(0 <= seg.left && seg.left <= seg.right && seg.right <= n - 1);
        }
        
        auto root = findSolution(segments, n);
        NCommon::validate(root, 0, n - 1);
        int evaluationCost = NCommon::evaluate(root, segments);
        return evaluationCost;
    }
}

void solve() {
    srand(322);
    dbl worst = 1.0;
    for (int i = 0; i < 100000; ++i) {
        int n = 1 + rand() % 20;
        auto segs = NCommon::generate(n, 1 + rand() % (50), 100000);
        int cubicCost = NCubic::getCost(n, segs);
        // assert(cubicCost == NNaive::getCost(n, segs));
        // int approxCost = NApprox::getCost(n, segs);
        int approxCost = NApproxExactBST::getCost(n, segs);
        assert(approxCost >= cubicCost);
        worst = max(worst, (dbl) approxCost / cubicCost);
    }
    cout << worst << endl;
}

signed main() {
#ifdef LOCAL
    // assert(freopen("input.txt", "r", stdin));
    // assert(freopen("output.txt", "w", stdout));
#endif
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << fixed << setprecision(20);

    int T = 1;
    // cin >> T;
    for (int i = 0; i < T; ++i) {
        solve();
    }

#ifdef LOCAL
    cout << endl << endl << "time = " << clock() / (double)CLOCKS_PER_SEC << endl;
#endif
}



