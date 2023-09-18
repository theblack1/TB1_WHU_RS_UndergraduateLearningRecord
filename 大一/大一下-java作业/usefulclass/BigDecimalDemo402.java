
package ch04.usefulclass;

import java.math.BigDecimal;

public class BigDecimalDemo402 {

	public static void main(String[] args) {

		// 创建BigDecimal，通过字符参数串创建
		BigDecimal number1 = new BigDecimal("999999999.99988888");
		// 创建BigDecimal，通过double参数创建
		BigDecimal number2 = new BigDecimal(567800000.888888);

		// 加法操作
		System.out.println("加法操作：" + number1.add(number2));
		// 减法操作
		System.out.println("减法操作：" + number1.subtract(number2));
		// 乘法操作
		System.out.println("乘法操作：" + number1.multiply(number2));
		// 除法操作
		System.out.println("除法操作：" + number1.divide(number2, BigDecimal.ROUND_HALF_UP));
	}
}
