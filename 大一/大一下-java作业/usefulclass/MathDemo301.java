
package ch04.usefulclass;

public class MathDemo301 {

	public static void main(String[] args) {

		double[] nums = { 1.4, 1.5, 1.6 };

		// 测试最大值和最小值
		System.out.printf("min(%.1f, %.1f) = %.1f\n", nums[1], nums[2], Math.min(nums[1], nums[2]));
		System.out.printf("max(%.1f, %.1f) = %.1f\n", nums[1], nums[2], Math.max(nums[1], nums[2]));
		System.out.println();

		// 测试三角函数
		// 1π弧度 = 180°
		System.out.printf("toDegrees(0.5π)	= %f\n", Math.toDegrees(0.5 * Math.PI));
		System.out.printf("toRadians(180/π) = %f\n", Math.toRadians(180 / Math.PI));
		System.out.println();

		// 测试平方根
		System.out.printf("sqrt(%.1f) = %f\n", nums[2], Math.sqrt(nums[2]));
		System.out.println();

		// 测试幂运算
		System.out.printf("pow(8, 3) = %f\n", Math.pow(8, 3));
		System.out.println();

		// 测试计算随机值
		System.out.printf("0.0~1.0之间的随机数 = %f\n", Math.random());
		System.out.println();

		// 测试舍入方法
		for (double num : nums) {
			display(num);
		}

	}

	// 测试舍入方法
	public static void display(double n) {
		System.out.printf("ceil(%.1f)	= %.1f\n", n, Math.ceil(n));
		System.out.printf("floor(%.1f) 	= %.1f\n", n, Math.floor(n));
		System.out.printf("round(%.1f) 	= %d\n", n, Math.round(n));
		System.out.println();
	}
}
