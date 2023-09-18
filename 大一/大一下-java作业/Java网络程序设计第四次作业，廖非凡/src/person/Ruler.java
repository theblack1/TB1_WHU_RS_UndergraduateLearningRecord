package person;

public class Ruler extends Person {
	public String level = "";

	public final static String L1 = "校长";
	public final static String L21 = "院长";
	public final static String L22 = "行政主管";
	public final static String L25 = "行政职员";
	public final static String L31 = "系主任";
	public final static String L32 = "实验中心主任";
	public final static String L33 = "研究所主任";
	public final static String L4 = "教师";

	public Ruler(String title, String name) {
		level = title;
		this.name = name;
		ShowMe();
	}

	public Ruler() {
	}

	@Override
	protected void Person(String name2) {
		super.name = name2;

	}

	@Override
	public void ShowMe() {
		System.out.println();
		System.out.println(level + ":" + name);

	}

}
