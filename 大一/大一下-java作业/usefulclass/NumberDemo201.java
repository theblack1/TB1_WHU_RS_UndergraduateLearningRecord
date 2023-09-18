
package ch04.usefulclass;

public class NumberDemo201 {

	public static void main(String[] args) {

		// 1.构造方法
		// 创建数值为80的Integer对象
		Integer objInt = new Integer(80);
		// 创建数值为80.0的Double对象
		Double objDouble = new Double(80.0);
		// 通过"80.0"字符串创建数值为80.0的Float对象
		Float objFloat = new Float("80.0");
		// 通过"80"字符串创建数值为80的Long对象
		Long objLong = new Long("80");

		// 2.Number类方法
		// Integer对象转换为long数值
		long longVar = objInt.longValue();
		// Double对象转换为int数值
		int intVar = objDouble.intValue();
		System.out.println("intVar = " + intVar);
		System.out.println("longVar = " + longVar);

		// 3.compareTo()方法
		Float objFloat2 = new Float(100);
		int result = objFloat.compareTo(objFloat2);
		// result = -1，表示objFloat小于objFloat2
		System.out.println(result);

		// 4.字符串转换为基本数据类型
		// 10进制"100"字符串转换为10进制数为100
		int intVar2 = Integer.parseInt("100");
		// 16进制"ABC"字符串转换为10进制数为2748
		int intVar3 = Integer.parseInt("ABC", 16);
		System.out.println("intVar2 = " + intVar2);
		System.out.println("intVar3 = " + intVar3);

		// 5.基本数据类型转换为字符串
		// 100转换为10进制字符串
		String str1 = Integer.toString(100);
		// 100转换为16进制字符串结果是64
		String str2 = Integer.toString(100, 16);
		System.out.println("str1 = " + str1);
		System.out.println("str2 = " + str2);

	}
}
