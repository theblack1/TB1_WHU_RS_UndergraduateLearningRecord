
package ch04.usefulclass;

public class BooleanDemo203 {

	public static void main(String[] args) {

		// 创建数值为true的Character对象true
		Boolean obj1 = new Boolean(true);
		// 通过字符串"true"创建Character对象true
		Boolean obj2 = new Boolean("true");
		// 通过字符串"True"创建Character对象true
		Boolean obj3 = new Boolean("True");
		// 通过字符串"TRUE"创建Character对象true
		Boolean obj4 = new Boolean("TRUE");
		// 通过字符串"false"创建Character对象false
		Boolean obj5 = new Boolean("false");
		// 通过字符串"Yes"创建Character对象false
		Boolean obj6 = new Boolean("Yes");
		// 通过字符串"abc"创建Character对象false
		Boolean obj7 = new Boolean("abc");

		System.out.println("obj1 = " + obj1);
		System.out.println("obj2 = " + obj2);
		System.out.println("obj3 = " + obj3);
		System.out.println("obj4 = " + obj4);
		System.out.println("obj5 = " + obj5);
		System.out.println("obj6 = " + obj6);
		System.out.println("obj7 = " + obj7);
	}
}
