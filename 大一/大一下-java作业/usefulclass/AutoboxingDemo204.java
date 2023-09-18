
package ch04.usefulclass;

public class AutoboxingDemo204 {

	public static void main(String[] args) {
		Integer objInt = new Integer(80);
		Double objDouble = new Double(80.0);
		// 自动拆箱
		double sum = objInt + objDouble;

		// 自动装箱
		// 自动装箱'C'转换为Character对象
		Character objChar = 'C';
		// 自动装箱true转换为Boolean对象
		Boolean objBoolean = true;
		// 自动装箱80.0f转换为Float对象
		Float objFloat = 80.0f;

		// 自动装箱100转换为Integer对象
		display(100);

		// 避免出现下面的情况
		Integer obj = null;
		int intVar = obj;// 运行期异常NullPointerException

	}

	/**
	 * @param objInt Integer对象
	 * @return int数值
	 */
	public static int display(Integer objInt) {

		System.out.println(objInt);

		// return objInt.intValue();
		// 自动拆箱Integer对象转换为int
		return objInt;
	}
}
