### 题目
实现一个异常安全的动态矩阵类 `Matrix`，支持强异常保证的 `resize` 操作，并提供移动赋值运算符和访问元素的方法。

### 简介
在 C++ 中，实现一个异常安全且高效的动态矩阵类是非常重要的。本题要求你实现一个 `Matrix` 类，该类能够动态调整大小，并且在调整大小时保证强异常安全。此外，还需要实现移动赋值运算符和访问元素的方法。

### 要求
1. **异常安全**：确保在操作过程中即使发生异常，对象也能保持一致的状态。
2. **强异常保证的 `resize` 操作**：在调整矩阵大小时，如果新矩阵的创建或数据复制失败，旧矩阵应保持不变。
3. **移动赋值运算符**：实现移动赋值运算符以提高性能。
4. **访问元素**：提供访问和修改矩阵元素的方法，并在索引越界时抛出异常。
5. **输出运算符**：实现格式化输出矩阵的方法。

### 实现工具
- **编程语言**：C++
- **标准库**：`<iostream>`, `<vector>`, `<memory>`, `<stdexcept>`, `<algorithm>`, `<utility>`

### 代码实现

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <utility>

class Matrix {
public:
    // 构造函数
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data_ = std::make_unique<std::unique_ptr<double[]>[]>(rows);
        for (size_t i = 0; i < rows; ++i) {
            data_[i] = std::make_unique<double[]>(cols);
        }
    }

    // 析构函数
    ~Matrix() = default;

    // 移动构造函数
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // 移动赋值运算符
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    // 强异常保证的 resize 操作
    void resize(size_t new_rows, size_t new_cols) {
        if (new_rows == rows_ && new_cols == cols_) {
            return; // 大小没变，直接返回
        }

        // 创建临时矩阵
        Matrix temp(new_rows, new_cols);

        // 复制尽可能多的元素
        size_t min_rows = std::min(rows_, new_rows);
        size_t min_cols = std::min(cols_, new_cols);

        for (size_t i = 0; i < min_rows; ++i) {
            for (size_t j = 0; j < min_cols; ++j) {
                temp(i, j) = (*this)(i, j);
            }
        }

        // 交换内容（不会抛出异常）
        swap(temp);
    }

    // 交换辅助函数
    void swap(Matrix& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
    }

    // 访问元素
    double& operator()(size_t row, size_t col) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[row][col];
    }

    const double& operator()(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[row][col];
    }

    // 获取行列数
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // 输出运算符友元声明
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

private:
    std::unique_ptr<std::unique_ptr<double[]>[]> data_;
    size_t rows_, cols_;
};

// 格式化输出运算符
std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "Matrix[" << matrix.rows_ << "][" << matrix.cols_ << "]:\n";
    os << std::fixed << std::setprecision(2);

    for (size_t i = 0; i < matrix.rows_; ++i) {
        for (size_t j = 0; j < matrix.cols_; ++j) {
            os << std::setw(8) << matrix(i, j);
        }
        os << '\n';
    }
    return os;
}

int main() {
    try {
        Matrix m(3, 3);
        m(0, 0) = 1.0;
        m(0, 1) = 2.0;
        m(0, 2) = 3.0;
        m(1, 0) = 4.0;
        m(1, 1) = 5.0;
        m(1, 2) = 6.0;
        m(2, 0) = 7.0;
        m(2, 1) = 8.0;
        m(2, 2) = 9.0;

        std::cout << "Original Matrix:\n" << m << std::endl;

        m.resize(4, 4);
        m(3, 3) = 10.0;

        std::cout << "Resized Matrix:\n" << m << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
```

### 代码说明
1. **构造函数**：
   - `Matrix` 构造函数接受行数和列数，并初始化二维动态数组 `data_`。

2. **析构函数**：
   - 默认析构函数，自动释放 `data_` 的内存。

3. **移动构造函数**：
   - 移动构造函数将 `other` 的资源转移给当前对象，并将 `other` 的状态重置。

4. **移动赋值运算符**：
   - 移动赋值运算符将 `other` 的资源转移给当前对象，并将 `other` 的状态重置。

5. **强异常保证的 `resize` 操作**：
   - `resize` 方法首先检查是否需要调整大小。
   - 创建一个临时矩阵 `temp`，并复制尽可能多的元素到 `temp`。
   - 使用 `swap` 方法交换当前矩阵和 `temp` 的内容，确保强异常保证。

6. **交换辅助函数**：
   - `swap` 方法交换两个矩阵的内容，不抛出异常。

7. **访问元素**：
   - `operator()` 提供对矩阵元素的访问和修改，索引越界时抛出 `std::out_of_range` 异常。

8. **获取行列数**：
   - `rows` 和 `cols` 方法分别返回矩阵的行数和列数。

9. **输出运算符**：
   - `operator<<` 用于格式化输出矩阵。

### 测试
- 编译并运行上述代码，观察输出的原始矩阵和调整大小后的矩阵。
- 尝试不同的矩阵大小和元素值，验证 `resize` 操作的正确性和异常安全性。
- 尝试访问越界的索引，验证异常处理是否正确。

通过这个题目，你可以深入理解 C++ 中的异常安全编程、动态内存管理以及自定义类的设计。