
package ch04.usefulclass;

import java.math.BigInteger;

public class BigIntegerDemo401 {

	public static void main(String[] args) {

		// 创建BigInteger，字符串表示10进制数值
		BigInteger number1 = new BigInteger("999999999999");
		// 创建BigInteger，字符串表示16进制数值
		BigInteger number2 = new BigInteger("567800000", 16);

		// 加法操作
		System.out.println("加法操作：" + number1.add(number2));
		// 减法操作
		System.out.println("减法操作：" + number1.subtract(number2));
		// 乘法操作
		System.out.println("乘法操作：" + number1.multiply(number2));
		// 除法操作
		System.out.println("除法操作：" + number1.divide(number2));
	}
}
